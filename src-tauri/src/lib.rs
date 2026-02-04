use opencv::{
    core,
    imgcodecs,
    imgproc,
    objdetect,
    prelude::*,
};
use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;

#[derive(serde::Serialize)]
struct FaceResult {
    base64: String,       // 切り抜き後の透過画像
    debug_base64: String, // 青枠と赤枠を描画した確認用画像
}

#[tauri::command]
fn process_face(path: String) -> Result<Vec<FaceResult>, String> { // 戻り値の型を変更
    println!("process_face() invoked: Debug Mode");

    opencv::core::set_use_optimized(true).ok();
    opencv::core::set_num_threads(0).ok();

    let img = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "画像の読み込みに失敗")?;

    let faces = detect_faces(&img)?;
    let faces_vec: Vec<core::Rect> = faces.iter().collect();

    if faces_vec.is_empty() {
        return Err("顔が検出されませんでした".to_string());
    }

    // 並列処理
    let results: Result<Vec<FaceResult>, String> = faces_vec.par_iter().map(|face| {
        let img_size = img.size().map_err(|e| e.to_string())?;

        // 1. キャンバス確保 (margin設定)
        let canvas_margin_top = (face.height as f32 * 1.0) as i32;     // 髪全体を含める
        let canvas_margin_bottom = (face.height as f32 * 0.2) as i32;  // 首は最小限
        let canvas_margin_side = (face.width as f32 * 0.3) as i32;     // 横髪も含める

        let canvas_x = (face.x - canvas_margin_side).max(0);
        let canvas_y = (face.y - canvas_margin_top).max(0);
        let canvas_w = (face.width + canvas_margin_side * 2).min(img_size.width - canvas_x);
        let canvas_h = (face.height + canvas_margin_top + canvas_margin_bottom).min(img_size.height - canvas_y);

        let canvas_rect = core::Rect::new(canvas_x, canvas_y, canvas_w, canvas_h);

        // 作業用画像 (Canvas) 切り出し
        let canvas_roi = core::Mat::roi(&img, canvas_rect).map_err(|e| e.to_string())?;
        let mut work_img = core::Mat::default();
        canvas_roi.copy_to(&mut work_img).map_err(|e| e.to_string())?;

        // 2. ヒント枠 (AI探索範囲) - 顔と髪の中心部分に限定
        let hint_margin_x = (face.width as f32 * 0.15) as i32;   // 左右から15%除外（耳周辺を除外）
        let hint_margin_top = (face.height as f32 * 0.05) as i32; // 上部から5%除外
        let hint_margin_bottom = (face.height as f32 * 0.3) as i32; // 下部から30%除外（顎を残しつつ首を除外）
        
        let hint_x = hint_margin_x;
        let hint_y = hint_margin_top;
        let hint_w = (work_img.cols() - hint_margin_x * 2).max(1);
        let hint_h = (work_img.rows() - hint_margin_top - hint_margin_bottom).max(1);
        let hint_rect = core::Rect::new(hint_x, hint_y, hint_w, hint_h);

        // --- ★ここが追加: デバッグ画像の作成 ---
        let mut debug_img = work_img.clone();

        // (A) 青い枠: 顔検出の結果 (Haar Cascade)
        // Global座標からCanvas相対座標に変換
        let face_rel_x = face.x - canvas_x;
        let face_rel_y = face.y - canvas_y;
        let face_rel_rect = core::Rect::new(face_rel_x, face_rel_y, face.width, face.height);
        
        imgproc::rectangle(
            &mut debug_img,
            face_rel_rect,
            core::Scalar::new(255.0, 0.0, 0.0, 0.0), // BGRなので青(255,0,0)
            2, imgproc::LINE_8, 0
        ).map_err(|e| e.to_string())?;

        // (B) 赤い枠: AIの探索範囲 (GrabCut Hint)
        imgproc::rectangle(
            &mut debug_img,
            hint_rect,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // BGRなので赤(0,0,255)
            2, imgproc::LINE_8, 0
        ).map_err(|e| e.to_string())?;

        // デバッグ画像のエンコード (JPEGで軽く済ます)
        let mut debug_buf = core::Vector::<u8>::new();
        imgcodecs::imencode(".jpg", &debug_img, &mut debug_buf, &core::Vector::new())
            .map_err(|e| e.to_string())?;
        let debug_base64 = general_purpose::STANDARD.encode(debug_buf.as_slice());

        // ------------------------------------

        // 3. マスク生成 (GrabCut)
        let mask = create_high_quality_mask(&work_img, hint_rect)?;

        // 4. 仕上げ処理
        let base64_img = apply_mask_and_encode_parallel(&work_img, &mask)?;

        // 結果をセットで返す
        Ok(FaceResult {
            base64: base64_img,
            debug_base64: debug_base64,
        })
    }).collect();

    results
}

fn detect_faces(img: &core::Mat) -> Result<core::Vector<core::Rect>, String> {
    let mut face_detector = objdetect::CascadeClassifier::new("haarcascade_frontalface_default.xml")
        .map_err(|_| "xmlファイルが見つかりません")?;

    // ここを修正
    let mut faces = core::Vector::<core::Rect>::new();
    
    face_detector.detect_multi_scale(
        img,
        &mut faces,
        1.1,
        5,
        0,
        core::Size::new(30, 30),
        core::Size::new(0, 0),
    ).map_err(|e| e.to_string())?;

    Ok(faces)
}

fn create_high_quality_mask(img: &core::Mat, rect: core::Rect) -> Result<core::Mat, String> {
    let mut mask = core::Mat::new_size_with_default(img.size().map_err(|e| e.to_string())?, core::CV_8UC1, core::Scalar::all(imgproc::GC_PR_BGD as f64)).map_err(|e| e.to_string())?;
    let mut bgd = core::Mat::default();
    let mut fgd = core::Mat::default();

    // 前処理: ノイズ除去とコントラスト調整（より強力に）
    let mut preprocessed = core::Mat::default();
    // バイラテラルフィルタでエッジを保持しつつノイズ除去
    imgproc::bilateral_filter(img, &mut preprocessed, 15, 100.0, 100.0, core::BORDER_DEFAULT).map_err(|e| e.to_string())?;
    
    // コントラスト強調（CLAHE）
    let mut lab = core::Mat::default();
    imgproc::cvt_color(&preprocessed, &mut lab, imgproc::COLOR_BGR2Lab, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
    
    let mut channels = core::Vector::<core::Mat>::new();
    core::split(&lab, &mut channels).map_err(|e| e.to_string())?;
    
    let l_channel = channels.get(0).map_err(|e| e.to_string())?;
    let mut clahe_output = core::Mat::default();
    let mut clahe = imgproc::create_clahe(2.0, core::Size::new(8, 8)).map_err(|e| e.to_string())?;
    clahe.apply(&l_channel, &mut clahe_output).map_err(|e| e.to_string())?;
    channels.set(0, clahe_output).map_err(|e| e.to_string())?;
    
    let mut enhanced_lab = core::Mat::default();
    core::merge(&channels, &mut enhanced_lab).map_err(|e| e.to_string())?;
    
    let mut enhanced = core::Mat::default();
    imgproc::cvt_color(&enhanced_lab, &mut enhanced, imgproc::COLOR_Lab2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;

    // GrabCut実行（反復回数を大幅に増やして精度最大化）
    imgproc::grab_cut(&enhanced, &mut mask, rect, &mut bgd, &mut fgd, 25, imgproc::GC_INIT_WITH_RECT).map_err(|e| e.to_string())?;

    let mut mask_fg = core::Mat::default();
    let mut mask_pr = core::Mat::default();
    core::compare(&mask, &core::Scalar::all(imgproc::GC_FGD as f64), &mut mask_fg, core::CMP_EQ).map_err(|e| e.to_string())?;
    core::compare(&mask, &core::Scalar::all(imgproc::GC_PR_FGD as f64), &mut mask_pr, core::CMP_EQ).map_err(|e| e.to_string())?;
    let mut bin = core::Mat::default();
    core::bitwise_or(&mask_fg, &mask_pr, &mut bin, &core::Mat::default()).map_err(|e| e.to_string())?;

    // モルフォロジー演算を強化してより精密に
    let mut temp = core::Mat::default();
    let k_open = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(5, 5), core::Point::new(-1, -1)).map_err(|e| e.to_string())?;
    imgproc::morphology_ex(&bin, &mut temp, imgproc::MORPH_OPEN, &k_open, core::Point::new(-1, -1), 2, core::BORDER_CONSTANT, core::Scalar::default()).map_err(|e| e.to_string())?;
    
    let mut closed = core::Mat::default();
    let k_close = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(7, 7), core::Point::new(-1, -1)).map_err(|e| e.to_string())?;
    imgproc::morphology_ex(&temp, &mut closed, imgproc::MORPH_CLOSE, &k_close, core::Point::new(-1, -1), 2, core::BORDER_CONSTANT, core::Scalar::default()).map_err(|e| e.to_string())?;
    
    // エッジを精緻化（距離変換で滑らかに）
    let mut dist = core::Mat::default();
    imgproc::distance_transform(&closed, &mut dist, imgproc::DIST_L2, 3, core::CV_32F).map_err(|e| e.to_string())?;
    
    // 距離変換を正規化
    let mut normalized = core::Mat::default();
    core::normalize(&dist, &mut normalized, 0.0, 255.0, core::NORM_MINMAX, core::CV_8U, &core::Mat::default()).map_err(|e| e.to_string())?;
    
    // 閾値処理でクリーンなマスクを作成
    let mut thresholded = core::Mat::default();
    imgproc::threshold(&normalized, &mut thresholded, 10.0, 255.0, imgproc::THRESH_BINARY).map_err(|e| e.to_string())?;
    
    // 元のマスクと組み合わせて最適化
    let mut final_mask = core::Mat::default();
    core::bitwise_and(&closed, &thresholded, &mut final_mask, &core::Mat::default()).map_err(|e| e.to_string())?;
    
    // ボカシは最小限に（髭の細部を保持）
    Ok(final_mask)
}

fn apply_mask_and_encode_parallel(img: &core::Mat, mask: &core::Mat) -> Result<String, String> {
    let width = img.cols() as usize;
    let height = img.rows() as usize;
    
    // 1. 空のMatを作成
    let mut final_mat = unsafe {
        core::Mat::new_rows_cols(
            height as i32,
            width as i32,
            core::CV_8UC4
        ).map_err(|e| e.to_string())?
    };

    // 2. メモリ領域を借りる
    let mat_data = final_mat.data_bytes_mut().map_err(|e| e.to_string())?;

    // 3. 並列書き込み
    mat_data.par_chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(y, row_slice)| {
            for x in 0..width {
                // ★修正箇所: .map(|p| *p) を使って「参照」ではなく「値」を取り出す
                // これで unwrap_or に一時変数の参照を渡す必要がなくなり、エラーが消える
                let color_pix = img.at_2d::<core::Vec3b>(y as i32, x as i32)
                    .map(|p| *p) // 参照から値へコピー
                    .unwrap_or(core::Vec3b::all(0)); // デフォルト値も値として渡す

                let alpha_val = mask.at_2d::<u8>(y as i32, x as i32)
                    .map(|p| *p) // 参照から値へコピー
                    .unwrap_or(0);

                let offset = x * 4;
                row_slice[offset + 0] = color_pix[0]; // B
                row_slice[offset + 1] = color_pix[1]; // G
                row_slice[offset + 2] = color_pix[2]; // R
                row_slice[offset + 3] = alpha_val;    // A (ここは *alpha_val じゃなくて alpha_val になる)
            }
        });

    // 4. エンコード
    let mut buf = core::Vector::<u8>::new();
    imgcodecs::imencode(".png", &final_mat, &mut buf, &core::Vector::new())
        .map_err(|e| e.to_string())?;

    Ok(general_purpose::STANDARD.encode(buf.as_slice()))
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[derive(serde::Serialize)]
struct FaceSwapResult {
    base64: String,  // 合成結果画像
    color_correction_strength: f64,  // 使用された色補正強度（0.0-1.0）
}

#[tauri::command]
fn face_swap(source_path: String, target_path: String, color_correction: Option<f64>) -> Result<FaceSwapResult, String> {
    opencv::core::set_use_optimized(true).ok();
    opencv::core::set_num_threads(0).ok();

    // 画像読み込み
    let source_img = imgcodecs::imread(&source_path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "ソース画像の読み込みに失敗")?;
    let target_img = imgcodecs::imread(&target_path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "ターゲット画像の読み込みに失敗")?;

    // 顔検出
    let source_faces = detect_faces(&source_img)?;
    let target_faces = detect_faces(&target_img)?;

    if source_faces.is_empty() {
        return Err("ソース画像に顔が検出されませんでした".to_string());
    }
    if target_faces.is_empty() {
        return Err("ターゲット画像に顔が検出されませんでした".to_string());
    }

    let source_face = source_faces.get(0).map_err(|e| e.to_string())?;
    let target_face = target_faces.get(0).map_err(|e| e.to_string())?;

    // ソース顔を検出矩形で切り抜き（顔だけ）
    let source_face_roi = core::Mat::roi(&source_img, source_face).map_err(|e| e.to_string())?;
    let mut source_face_img = core::Mat::default();
    source_face_roi.copy_to(&mut source_face_img).map_err(|e| e.to_string())?;

    // ターゲット顔も切り抜き（色補正用）
    let target_face_roi = core::Mat::roi(&target_img, target_face).map_err(|e| e.to_string())?;
    let mut target_face_img = core::Mat::default();
    target_face_roi.copy_to(&mut target_face_img).map_err(|e| e.to_string())?;

    // 色補正強度を自動計算（肌色の差に基づく）
    let auto_correction_strength = if color_correction.is_none() {
        calculate_color_correction_strength(&source_face_img, &target_face_img)?
    } else {
        color_correction.unwrap()
    };

    // ソース顔をターゲット顔のサイズにリサイズ
    let mut resized_face = core::Mat::default();
    imgproc::resize(
        &source_face_img, 
        &mut resized_face, 
        core::Size::new(target_face.width, target_face.height), 
        0.0, 0.0, 
        imgproc::INTER_LANCZOS4
    ).map_err(|e| e.to_string())?;

    // 色補正: ソース顔の色をターゲット顔に合わせる
    let mut color_corrected = core::Mat::default();
    match_color(&resized_face, &target_face_img, &mut color_corrected, auto_correction_strength)?;

    // 楽円マスクを作成（顔全体を滑らかに合成）
    let mask = create_ellipse_mask(target_face.width, target_face.height)?;

    // ターゲット画像のコピーを作成
    let mut result = target_img.clone();

    // ターゲット顔の位置に直接ブレンド
    blend_with_mask(&color_corrected, &mut result, &mask, target_face.x, target_face.y)?;

    // エンコード
    let mut buf = core::Vector::<u8>::new();
    imgcodecs::imencode(".png", &result, &mut buf, &core::Vector::new())
        .map_err(|e| e.to_string())?;

    Ok(FaceSwapResult {
        base64: general_purpose::STANDARD.encode(buf.as_slice()),
        color_correction_strength: auto_correction_strength,
    })
}

fn extract_face_with_mask(img: &core::Mat, face: &core::Rect) -> Result<(core::Mat, core::Mat), String> {
    let img_size = img.size().map_err(|e| e.to_string())?;

    // キャンバス確保
    let canvas_margin_top = (face.height as f32 * 1.0) as i32;
    let canvas_margin_bottom = (face.height as f32 * 0.2) as i32;
    let canvas_margin_side = (face.width as f32 * 0.3) as i32;

    let canvas_x = (face.x - canvas_margin_side).max(0);
    let canvas_y = (face.y - canvas_margin_top).max(0);
    let canvas_w = (face.width + canvas_margin_side * 2).min(img_size.width - canvas_x);
    let canvas_h = (face.height + canvas_margin_top + canvas_margin_bottom).min(img_size.height - canvas_y);

    let canvas_rect = core::Rect::new(canvas_x, canvas_y, canvas_w, canvas_h);
    let canvas_roi = core::Mat::roi(img, canvas_rect).map_err(|e| e.to_string())?;
    let mut face_img = core::Mat::default();
    canvas_roi.copy_to(&mut face_img).map_err(|e| e.to_string())?;

    // ヒント枠
    let hint_margin_x = (face.width as f32 * 0.15) as i32;
    let hint_margin_top = (face.height as f32 * 0.05) as i32;
    let hint_margin_bottom = (face.height as f32 * 0.15) as i32;  // 顧を含めるたも30%から15%に減少
    
    let hint_x = hint_margin_x;
    let hint_y = hint_margin_top;
    let hint_w = (face_img.cols() - hint_margin_x * 2).max(1);
    let hint_h = (face_img.rows() - hint_margin_top - hint_margin_bottom).max(1);
    let hint_rect = core::Rect::new(hint_x, hint_y, hint_w, hint_h);

    // マスク生成
    let mask = create_high_quality_mask(&face_img, hint_rect)?;

    Ok((face_img, mask))
}

// 楽円マスクを作成（face swap用）
fn create_ellipse_mask(width: i32, height: i32) -> Result<core::Mat, String> {
    let mut mask = core::Mat::new_size_with_default(
        core::Size::new(width, height),
        core::CV_8UC1,
        core::Scalar::all(0.0)
    ).map_err(|e| e.to_string())?;

    let center = core::Point::new(width / 2, height / 2);
    // 楕円を縦長にして髪への侵食を防ぐ（横幅を狭く、縦はそのまま）
    let axes = core::Size::new((width as f32 * 0.40) as i32, (height as f32 * 0.48) as i32);
    
    // 白い楽円を描画
    imgproc::ellipse(
        &mut mask,
        center,
        axes,
        0.0,
        0.0,
        360.0,
        core::Scalar::all(255.0),
        -1,
        imgproc::LINE_8,
        0
    ).map_err(|e| e.to_string())?;

    // エッジを最小限にぼかす（境界を自然に）
    let mut smooth_mask = core::Mat::default();
    imgproc::gaussian_blur(
        &mask,
        &mut smooth_mask,
        core::Size::new(5, 5),
        1.5,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT
    ).map_err(|e| e.to_string())?;

    Ok(smooth_mask)
}

// 肌色の差に基づいて色補正強度を自動計算
fn calculate_color_correction_strength(src: &core::Mat, target: &core::Mat) -> Result<f64, String> {
    // 肌色を抽出（YCrCbカラースペース使用）
    let mut src_ycrcb = core::Mat::default();
    let mut target_ycrcb = core::Mat::default();
    imgproc::cvt_color(src, &mut src_ycrcb, imgproc::COLOR_BGR2YCrCb, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
    imgproc::cvt_color(target, &mut target_ycrcb, imgproc::COLOR_BGR2YCrCb, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;

    // 肌色範囲でマスクを作成clahe
    let mut src_skin_mask = core::Mat::default();
    let mut target_skin_mask = core::Mat::default();
    let lower_skin = core::Scalar::new(0.0, 133.0, 77.0, 0.0);
    let upper_skin = core::Scalar::new(255.0, 173.0, 127.0, 0.0);
    
    core::in_range(&src_ycrcb, &lower_skin, &upper_skin, &mut src_skin_mask).map_err(|e| e.to_string())?;
    core::in_range(&target_ycrcb, &lower_skin, &upper_skin, &mut target_skin_mask).map_err(|e| e.to_string())?;

    // 肌色領域の平均色を計算
    let src_skin_mean = core::mean(src, &src_skin_mask).map_err(|e| e.to_string())?;
    let target_skin_mean = core::mean(target, &target_skin_mask).map_err(|e| e.to_string())?;

    // 色の差を計算（ユークリッド距離）
    let color_diff = (
        (target_skin_mean[0] - src_skin_mean[0]).powi(2) +
        (target_skin_mean[1] - src_skin_mean[1]).powi(2) +
        (target_skin_mean[2] - src_skin_mean[2]).powi(2)
    ).sqrt();

    // 色差に基づいて補正強度を決定（差が大きいほど強く補正）
    // 双曲線関数を使用: strength = max_strength * color_diff / (color_diff + k)
    // これにより滑らかな曲線で強度が増加し、最終的に飽和する
    let max_strength = 0.7;  // 最大強度70%
    let k = 40.0;  // この値で強度カーブの傾きを調整（色差40で約半分の強度）
    
    let strength = max_strength * color_diff / (color_diff + k);

    Ok(strength)
}

// 色補正: ソース画像の肌色をターゲット画像の肌色に合わせる
fn match_color(src: &core::Mat, target: &core::Mat, dst: &mut core::Mat, strength: f64) -> Result<(), String> {
    // 肌色を抽出（YCrCbカラースペース使用）
    let mut src_ycrcb = core::Mat::default();
    let mut target_ycrcb = core::Mat::default();
    imgproc::cvt_color(src, &mut src_ycrcb, imgproc::COLOR_BGR2YCrCb, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
    imgproc::cvt_color(target, &mut target_ycrcb, imgproc::COLOR_BGR2YCrCb, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;

    // 肌色範囲でマスクを作成（Cr: 133-173, Cb: 77-127 あたりが肌色）
    let mut src_skin_mask = core::Mat::default();
    let mut target_skin_mask = core::Mat::default();
    let lower_skin = core::Scalar::new(0.0, 133.0, 77.0, 0.0);
    let upper_skin = core::Scalar::new(255.0, 173.0, 127.0, 0.0);
    
    core::in_range(&src_ycrcb, &lower_skin, &upper_skin, &mut src_skin_mask).map_err(|e| e.to_string())?;
    core::in_range(&target_ycrcb, &lower_skin, &upper_skin, &mut target_skin_mask).map_err(|e| e.to_string())?;

    // 肌色領域の平均色を計算
    let src_skin_mean = core::mean(src, &src_skin_mask).map_err(|e| e.to_string())?;
    let target_skin_mean = core::mean(target, &target_skin_mask).map_err(|e| e.to_string())?;

    // 肌色の差分を計算（strengthで補正強度を調整）
    let color_shift = [
        (target_skin_mean[0] - src_skin_mean[0]) * strength,
        (target_skin_mean[1] - src_skin_mean[1]) * strength,
        (target_skin_mean[2] - src_skin_mean[2]) * strength,
    ];

    // 変換後の画像を作成
    src.copy_to(dst).map_err(|e| e.to_string())?;
    
    // 色差分を加算（輝度補正は削除して色シフトのみ）
    let mut dst_f32 = core::Mat::default();
    dst.convert_to(&mut dst_f32, core::CV_32F, 1.0, 0.0).map_err(|e| e.to_string())?;
    
    let scalar_shift = core::Scalar::new(color_shift[0], color_shift[1], color_shift[2], 0.0);
    let mut shifted = core::Mat::default();
    core::add(&dst_f32, &scalar_shift, &mut shifted, &core::Mat::default(), -1).map_err(|e| e.to_string())?;
    
    shifted.convert_to(dst, core::CV_8U, 1.0, 0.0).map_err(|e| e.to_string())?;

    Ok(())
}

fn blend_with_mask(src: &core::Mat, dst: &mut core::Mat, mask: &core::Mat, x: i32, y: i32) -> Result<(), String> {
    let height = src.rows();
    let width = src.cols();

    for row in 0..height {
        for col in 0..width {
            let dst_y = y + row;
            let dst_x = x + col;

            if dst_y >= 0 && dst_y < dst.rows() && dst_x >= 0 && dst_x < dst.cols() {
                let alpha = *mask.at_2d::<u8>(row, col).map_err(|e| e.to_string())? as f32 / 255.0;
                let src_pixel = src.at_2d::<core::Vec3b>(row, col).map_err(|e| e.to_string())?;
                let dst_pixel = dst.at_2d_mut::<core::Vec3b>(dst_y, dst_x).map_err(|e| e.to_string())?;

                for c in 0..3 {
                    dst_pixel[c] = (src_pixel[c] as f32 * alpha + dst_pixel[c] as f32 * (1.0 - alpha)) as u8;
                }
            }
        }
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![greet, process_face, face_swap])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
