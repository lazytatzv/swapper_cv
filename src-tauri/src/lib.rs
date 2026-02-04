use opencv::{
    core,
    imgcodecs,
    imgproc,
    objdetect,
    prelude::*,
    types,
};

use base64::{Engine as _, engine::general_purpose};

#[tauri::command]
fn process_face(path: String) -> Result<String, String> {
    println!("process_face() invoked!");
    
    let img = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "画像の読み込みに失敗")?;

    let mut face_detector = objdetect::CascadeClassifier::new("haarcascade_frontalface_default.xml")
        .map_err(|_| "xmlファイルがないよ")?;

    let mut faces = types::VectorOfRect::new();

    face_detector.detect_multi_scale(
        &img,
        &mut faces,
        1.1,
        3,
        0,
        core::Size::new(30, 30),
        core::Size::new(0, 0),
    ).map_err(|e| e.to_string())?;

    for face in faces.iter() {
        // --- 改善点1: マージン（余白）を追加 ---
        // 顔の検出枠より少し広めに切り抜く準備
        let margin = (face.width as f32 * 0.3) as i32; // 30%広げる
        let img_size = img.size().map_err(|e| e.to_string())?;

        // 画像からはみ出さないように座標計算
        let safe_x = (face.x - margin).max(0);
        let safe_y = (face.y - margin).max(0);
        let safe_w = (face.width + margin * 2).min(img_size.width - safe_x);
        let safe_h = (face.height + margin * 2).min(img_size.height - safe_y);

        let loose_face_rect = core::Rect::new(safe_x, safe_y, safe_w, safe_h);
        
        // GrabCut用のRectは、広げた領域の中で、元の顔の位置を指定する必要がある
        // ただし、今回はシンプルに「広げた領域全体」に対して処理を行い、
        // 「元の顔枠」をヒント（前景候補）として渡す戦略をとります

        // 1. GrabCut用マスク
        let mut mask = core::Mat::new_size_with_default(
            img.size().map_err(|e| e.to_string())?,
            core::CV_8UC1,
            core::Scalar::all(imgproc::GC_PR_BGD as f64), // 背景候補
        ).map_err(|e| e.to_string())?;

        let mut bgd_model = core::Mat::default();
        let mut fgd_model = core::Mat::default();

        // 2. GrabCut実行 (今回は広げたloose_face_rectではなく、正確なfaceを使う)
        // faceの領域を「たぶん前景」として初期化
        imgproc::grab_cut(
            &img,
            &mut mask,
            face, // 検出された顔の枠を使う
            &mut bgd_model,
            &mut fgd_model,
            5, // 反復回数
            imgproc::GC_INIT_WITH_RECT,
        ).map_err(|e| e.to_string())?;

        // 3. マスクの生成 ("確実な前景"と"たぶん前景"を抽出)
        let mut final_mask = core::Mat::default();
        // 前景(1) または たぶん前景(3) を抽出
        // (GC_FGD = 1, GC_PR_FGD = 3)
        // ここでは単純化のため、マスク値が1か3の場所を255(白)、それ以外を0(黒)にする処理
        let mut bin_mask = core::Mat::default();
        
        // 論理演算でマスクを作る: (mask & 1) * 255
        // これで前景(1)とたぶん前景(3)の最下位ビットが1なので抽出できるトリック
        // RustのOpenCVラッパーだと少し面倒なので、compareを使うのが無難
        
        let mut mask_fg = core::Mat::default();
        let mut mask_pr_fg = core::Mat::default();
        
        core::compare(&mask, &core::Scalar::all(imgproc::GC_FGD as f64), &mut mask_fg, core::CMP_EQ).map_err(|e| e.to_string())?;
        core::compare(&mask, &core::Scalar::all(imgproc::GC_PR_FGD as f64), &mut mask_pr_fg, core::CMP_EQ).map_err(|e| e.to_string())?;
        
        core::bitwise_or(&mask_fg, &mask_pr_fg, &mut bin_mask, &core::Mat::default()).map_err(|e| e.to_string())?;

        // --- 改善点2: マスクの境界をぼかす（フェザー処理） ---
        let mut blurred_mask = core::Mat::default();
        imgproc::gaussian_blur(
            &bin_mask,
            &mut blurred_mask,
            core::Size::new(9, 9), // ぼかし強度 (奇数)
            0.0,
            0.0,
            core::BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        ).map_err(|e| e.to_string())?;

        // 4. BGRA変換
        let mut bgra = core::Mat::default();
        imgproc::cvt_color(
            &img,
            &mut bgra,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_APPROX
        ).map_err(|e| e.to_string())?;

        // 5. アルファチャンネルにぼかしたマスクを適用
        
        // まず「窓（参照）」を作ります
        let roi_bgra_ref = core::Mat::roi(&bgra, loose_face_rect).map_err(|e| e.to_string())?;
        
        // ★ここが重要！ clone() でコピーして、書き換え可能な「実体」にします
        let mut final_face = roi_bgra_ref.try_clone().map_err(|e| e.to_string())?;

        let roi_mask = core::Mat::roi(&blurred_mask, loose_face_rect).map_err(|e| e.to_string())?;

        // ROI内でアルファチャンネル書き換え
        for y in 0..final_face.rows() {
            for x in 0..final_face.cols() {
                // マスクの明るさ(0-255)を取得
                let alpha_val = *roi_mask.at_2d::<u8>(y, x).map_err(|e| e.to_string())?;
                
                // ★ final_face なら at_2d_mut（書き込み）が使えます！
                let pixel = final_face.at_2d_mut::<core::Vec4b>(y, x)
                    .map_err(|e: opencv::Error| e.to_string())?;
                pixel[3] = alpha_val; // Alpha値をマスクの明るさにする
            }
        }

        // 6. エンコードして返却 (final_face を渡す)
        let mut buf = core::Vector::<u8>::new();
        imgcodecs::imencode(".png", &final_face, &mut buf, &core::Vector::new())
            .map_err(|e| e.to_string())?;
            
        return Ok(general_purpose::STANDARD.encode(buf.as_slice()));

    }

    Err("顔が検出されませんでした".to_string())
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
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
