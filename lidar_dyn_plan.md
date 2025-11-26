# LiDAR動的解像度対応 (SACのみ)

## 目的
- `--use_lidar_dyn`フラグを追加し、ONのときだけLiDARビーム本数を動的に切り替える。
- 直前ステップ(n-1)の最小障害物距離が5m超なら次ステップ(n)は12本、それ以下なら従来の120本。
- ネットワーク入力次元は常に120のまま保持。
- ロギングするビーム本数指標も実際に使った本数（12/120）を記録。
- フラグOFF時は従来と完全同一挙動。

## 想定変更ファイル
- `src/train/train_HOPE_sac.py`
  - argparseに`--use_lidar_dyn`を追加（デフォルトFalse）。
  - 環境生成時にフラグを渡す。
  - LiDARビーム本数のロギングを「実際に使った本数」を参照するように変更（情報の受け渡し仕様に合わせる）。
- `src/env/car_parking_base.py`
  - 動的LiDARフラグ/状態を受け取り保持。
  - 直前ステップの最小距離を記録し、次ステップのLiDAR解像度を決定。
  - LiDAR観測取得処理を、実使用ビーム数とネットワーク入力（常に120次元）に分離。
  - `info`や観測に「今回使ったビーム本数」を伝搬するためのメタデータを追加。
- `src/env/lidar_simulator.py`
  - 12本/120本の両方のビームセットを扱えるように拡張（生成済みビームを再利用 or サブセット化）。
  - 12本モード時も最終的な観測ベクトル長を120に戻す（補間/複製などの方法を実装）。
  - 車体境界オフセット計算も12本モード対応。
- `src/env/env_wrapper.py`
  - LiDARメタデータ（実ビーム数）を観測/返り値に保持する場合、その受け渡しと`observation_shape`への影響がゼロになるよう調整。
- `src/evaluation/eval_utils.py`
  - LiDARビーム本数の記録を、実際に使った本数（メタデータ）に基づくよう変更（SACのダイナミックモードに対応）。

## 実施内容まとめ
- `--use_lidar_dyn`（SACのみ）を追加。デフォルトOFFで従来挙動維持。
- `car_parking_base.py`: 直前ステップ最小距離>5mなら次ステップは12本、それ以外は120本。使用本数は`info['lidar_beam_used']`で通知。初回ステップは120本。
- `lidar_simulator.py`: 任意ビーム数で取得し、12本取得時は角度方向の線形補間で120次元にアップサンプリングし観測形状固定。ビームセット/車体オフセットはビーム数ごとにキャッシュ。
- `train_HOPE_sac.py` / `evaluation/eval_utils.py`: ログのビーム本数に`info['lidar_beam_used']`を利用。
- PPO版は未変更のまま。

## 追加で走らせたいチェック
- `python src/train/train_HOPE_sac.py --train_episode 1 --use_lidar_dyn` を短縮実行し、ログにビーム本数が12/120で切り替わることを確認。
- `python src/evaluation/eval_utils.py`（SAC用の呼び出しパラメータに合わせて）で記録が壊れていないか軽く回す。
