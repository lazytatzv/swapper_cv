use opencv::{
    core,
    imgcodecs,
    imgproc,
    objdetect,
    prelude::*,
    //types,
};
use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;

#[tauri::command]
fn process_face(path: String) -> Result<Vec<String>, String> {
    println!("process_face() invoked: Single-Image Multi-Core Attack");

    // --- ★1. OpenCVに「持てる力の全てを出せ」と命令する ---
    // これでGrabCut内部の行列計算などが（可能な範囲で）並列化されます
    opencv::core::set_use_optimized(true).ok();
    opencv::core::set_num_threads(0).ok(); // 0 = システムの全コアを使う

    // 1. 画像読み込み
    let img = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "画像の読み込みに失敗")?;

    let faces = detect_faces(&img)?;
    let faces_vec: Vec<core::Rect> = faces.iter().collect();

    if faces_vec.is_empty() {
        return Err("顔が検出されませんでした".to_string());
    }

    // ここは複数人の並列処理（前回と同じ）
    let results: Result<Vec<String>, String> = faces_vec.par_iter().map(|face| {
        let img_size = img.size().map_err(|e| e.to_string())?;

        // --- キャンバス確保 ---
        let canvas_margin_top = (face.height as f32 * 1.0) as i32;
        let canvas_margin_bottom = (face.height as f32 * 0.4) as i32;
        let canvas_margin_side = (face.width as f32 * 0.3) as i32;

        let canvas_x = (face.x - canvas_margin_side).max(0);
        let canvas_y = (face.y - canvas_margin_top).max(0);
        let canvas_w = (face.width + canvas_margin_side * 2).min(img_size.width - canvas_x);
        let canvas_h = (face.height + canvas_margin_top + canvas_margin_bottom).min(img_size.height - canvas_y);

        let canvas_rect = core::Rect::new(canvas_x, canvas_y, canvas_w, canvas_h);

        // 作業用画像 (Canvas) 切り出し
        let canvas_roi = core::Mat::roi(&img, canvas_rect).map_err(|e| e.to_string())?;
        let mut work_img = core::Mat::default();
        canvas_roi.copy_to(&mut work_img).map_err(|e| e.to_string())?;

        // --- ヒント枠 ---
        let border = 2;
        let hint_w = (work_img.cols() - border * 2).max(1);
        let neck_exclude_px = (face.height as f32 * 0.4) as i32; 
        let hint_h = (work_img.rows() - border - neck_exclude_px).max(1);
        let hint_rect = core::Rect::new(border, border, hint_w, hint_h);

        // --- マスク生成 (GrabCut) ---
        // ※ここはアルゴリズムの性質上、どうしても1コア集中になりがちですが、
        // 冒頭の set_num_threads(0) が効く部分は効きます。
        let mask = create_high_quality_mask(&work_img, hint_rect)?;

        // --- ★2. 最後の仕上げ処理を全コアで殴る ---
        // apply_mask_and_encode_parallel という新関数を作りました
        let base64_img = apply_mask_and_encode_parallel(&work_img, &mask)?;

        Ok(base64_img)
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

    // GrabCut実行
    imgproc::grab_cut(img, &mut mask, rect, &mut bgd, &mut fgd, 5, imgproc::GC_INIT_WITH_RECT).map_err(|e| e.to_string())?;

    let mut mask_fg = core::Mat::default();
    let mut mask_pr = core::Mat::default();
    core::compare(&mask, &core::Scalar::all(imgproc::GC_FGD as f64), &mut mask_fg, core::CMP_EQ).map_err(|e| e.to_string())?;
    core::compare(&mask, &core::Scalar::all(imgproc::GC_PR_FGD as f64), &mut mask_pr, core::CMP_EQ).map_err(|e| e.to_string())?;
    let mut bin = core::Mat::default();
    core::bitwise_or(&mask_fg, &mask_pr, &mut bin, &core::Mat::default()).map_err(|e| e.to_string())?;

    let mut smooth = core::Mat::default();
    let k_open = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(3, 3), core::Point::new(-1, -1)).map_err(|e| e.to_string())?;
    let mut temp = core::Mat::default();
    imgproc::morphology_ex(&bin, &mut temp, imgproc::MORPH_OPEN, &k_open, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, core::Scalar::default()).map_err(|e| e.to_string())?;
    let k_close = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(9, 9), core::Point::new(-1, -1)).map_err(|e| e.to_string())?;
    imgproc::morphology_ex(&temp, &mut smooth, imgproc::MORPH_CLOSE, &k_close, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, core::Scalar::default()).map_err(|e| e.to_string())?;
    Ok(smooth)
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![greet, process_face])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
