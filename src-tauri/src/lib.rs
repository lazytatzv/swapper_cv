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
    
    let mut img = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)
        .map_err(|_| "画像の読み込みに失敗")?;

    let mut face_detector = objdetect::CascadeClassifier::new("haarcascade_frontalface_default.xml")
            .map_err(|_| "xmlファイルがないよ")?;

    let mut faces = types::VectorOfRect::new();

    face_detector.detect_multi_scale(
        &img,
        &mut faces,
        1.1, // scaleFactor. 小さくすると制度が上がるが重くなる
        3, // minNeighbors 値をあげるとご検出が減るが、本物を見逃しやすくなる
        0, // flags. 0でいい.
        core::Size::new(30, 30),// minSize
        core::Size::new(0, 0),  //maxSize
    ).map_err(|e| e.to_string())?;

    println!("the function is running");


    for face in faces.iter() {
        imgproc::rectangle(
            &mut img,
            face, //描く範囲
            core::Scalar::new(255.0, 255.0, 0.0, 0.0), // 色
            2, // 線の太さpixel
            imgproc::LINE_8, // 線の種類. ここでは標準
            0, //shift
        ).map_err(|e| e.to_string())?;
    }

    // RustのVec<T>だとC++がわからない
    let mut buf = core::Vector::<u8>::new();
    imgcodecs::imencode(".png", &img, &mut buf, &core::Vector::new())
        .map_err(|e| e.to_string())?;

    println!("func finished");
    Ok(general_purpose::STANDARD.encode(buf.as_slice()))
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
