import { useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/core";
import { open } from '@tauri-apps/plugin-dialog';
import "./App.css";

function App() {
  const [greetMsg, setGreetMsg] = useState("");
  const [name, setName] = useState("");


  const [imgData, setImgData] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const selectAndProcess = async () => {
    const file = await open({
      multiple: false, // 一枚だけ。string
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
    });

    console.log("Selected:", file);

    if (file && typeof file === 'string') {
      setLoading(true);
      try {
        const base64: string = await invoke("process_face", { path: file });
        setImgData(`data:image/png;base64,${base64}`);
        console.log("Ok. try block");
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
  };

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
    setGreetMsg(await invoke("greet", { name }));
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
        Face Swapper Next
      </h1>

      <div 
        onClick={selectAndProcess}
        className="w-full max-w-2xl aspect-video border-2 border-dashed border-slate-700 rounded-3xl flex flex-col items-center justify-center cursor-pointer hover:border-cyan-500 transition-all bg-slate-800/50 backdrop-blur-sm overflow-hidden"
      >
        {imgData ? (
          <img src={imgData} className="w-full h-full object-contain" />
        ) : (
          <div className="text-center">
            <p className="text-slate-400 text-lg">{loading ? "処理中..." : "画像を選択して開始"}</p>
          </div>
        )}
      </div>

      <div className="mt-8 flex gap-4">
        {/* ここにコントラスト調整のスライダーとかを後で追加 */}
        <button 
          onClick={selectAndProcess}
          className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-full font-semibold transition-colors shadow-lg shadow-cyan-900/20"
        >
          画像を読み込む
        </button>
      </div>
    </div>
  );
}

export default App;
