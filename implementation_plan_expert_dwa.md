# 実装計画: LiDAR危険時のA*+DWA専門家行動収集とSAC損失拡張

## 目的
LiDARが危険（1m未満）を検知したときに専門家（A*+DWA）による脱出行動を実行し、RL行動と専門家行動の両方をリプレイバッファに記録。サンプルに専門家行動が含まれる場合、SACのactor損失にMSEを加えて模倣学習を併用する。
CLI実行時に `--use_expert_il` フラグでこの機能（専門家利用＋模倣損失）をオン/オフ切替できるようにする。

## 大まかな手順
1) LiDAR最小距離による危険判定を追加し、危険時は実行を専門家プランナに切替える。  
2) `a_dwa.md` に沿った A*（グローバル）+ DWA（ローカル）の融合プランナを実装し、`plan_action(state, map)` -> `(steer, speed)` を提供。  
3) リプレイバッファのサンプルにオプションの `expert_action` を追加し、危険時はRL行動と専門家行動の両方を保存。  
4) SACのupdateで、`expert_action` を含むサンプルに対して actor損失へ MSE(actor_action, expert_action) を λ=1 で加算。  
5) 専門家利用のメトリクス/ログを追加し、トレーニングループが動くことを確認。

## 触る予定のファイル
- `src/env/car_parking_base.py` または `src/env/env_wrapper.py`: LiDARの最小距離をinfoなどで外に出し、訓練ループが危険判定できるようにする。  
- `src/model/agent/parking_agent.py`: プランナ使用時にRL行動と専門家行動を保持して保存できるようにする。  
- `src/model/agent/sac_agent.py`: リプレイバッファのスキーマを `expert_action` 付きに拡張し、actor損失分岐でMSEを追加。  
- `src/model/replay_memory.py`: オプションの `expert_action` フィールドを許可。  
- `src/train/train_HOPE_sac.py`: 観測/情報から危険を検出し、危険時は専門家プランナで実行、両方の行動をメモリに記録し `push_memory` に渡す。  
- `src/model/planner/a_dwa_fusion.py`（新規）: A* + DWA 融合を実装し、1ステップの `(steer, speed)` を返す。内部でサブゴールを保持。

## データ/制御フロー
1. `env.step` がLiDAR入り観測を返す。訓練ループで `min(lidar) < 1.0` なら `danger=True`。  
2. 危険でなければ従来どおりRL行動を使用し、`expert_action=None` を保存。  
3. 危険なら:  
   - プランナから `expert_action` を取得。  
   - 環境は `expert_action` で進める（RL行動ではない）。  
   - `(obs, rl_action, reward, done, log_prob_rl, next_obs, expert_action)` をリプレイに保存。  
4. `SACAgent.update` でバッチをサンプル:  
   - 標準のSAC critic/actor損失を計算。  
   - `expert_action` があるサンプルでは `mse(actor_action, expert_action)` を計算し、actor損失に加算（λ=1）。  
   - critic/alpha更新は従来どおり。

## プランナ詳細（`a_dwa.md` より）
- グローバル: マンハッタンヒューリスティックと動的重み `f=g+exp(P)*h` を用いる改良A*。まずは8近傍で実装し、方向制限はTODOでも可。  
- 経路簡略化: 曲がり角/一定間隔のキーノードを抽出し、DWAのサブゴール列にする。  
- ローカル: `G = 15*vel + 37*head + 0.02*dist` の評価でDWA。運動学制約を守り、最良の (v, w) を選択。  
- 出力: (v, w) を環境のアクションスケールに合う `(steer, speed)` に変換。

## ハイパラ・定数
- 危険閾値: LiDAR最小距離 < 1.0 m。  
- 模倣重み λ: 1.0。  
- プランナ/DWAパラメータ: 速度・舵角は環境の上限を使用。動的ウィンドウは `a_dwa.md` に従い、必要なら VALID_SPEED/VALID_STEER と加速度上限から算出。

## テスト・検証
- ミニチェック: リプレイメモリが `expert_action` を保持でき、専門家なしでもSAC更新が動くことを小スクリプトで確認。  
- スモーク: 短い訓練（少数エピソード、visualize=False）でクラッシュせず、強制危険時に専門家分岐が走ることを確認。  
- ログ: 専門家サンプル数などをTensorBoardやstdoutに出力。
