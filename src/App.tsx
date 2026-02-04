import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from '@tauri-apps/plugin-dialog';
import "./App.css";

// Rustから返ってくるデータの型定義
interface FaceResult {
  base64: string;       // 切り抜き後の画像 (PNG)
  debug_base64: string; // 解析用画像 (JPG: 赤枠・青枠付き)
}

function App() {
  const [results, setResults] = useState<FaceResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const selectAndProcess = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
    });

    console.log("Selected:", file);

    if (file && typeof file === 'string') {
      setLoading(true);
      setResults([]); // 前の結果をクリア
      try {
        // Rustコマンド呼び出し (戻り値は FaceResult の配列)
        const data = await invoke<FaceResult[]>("process_face", { path: file });
        console.log("Processed faces:", data.length);
        setResults(data);
      } catch (e) {
        console.error(e);
        alert("エラーが発生しました: " + e);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center p-8">
      <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
        Face Swapper Debug
      </h1>

      {/* --- コントロールエリア --- */}
      <div className="mb-8 flex gap-4">
        <button 
          onClick={selectAndProcess}
          disabled={loading}
          className="px-8 py-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 rounded-full font-bold text-lg transition-all shadow-lg shadow-cyan-900/20 flex items-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              処理中...
            </>
          ) : (
            "画像を選択して解析"
          )}
        </button>
      </div>

      {/* --- 結果表示エリア --- */}
      <div className="w-full max-w-6xl space-y-12">
        
        {/* 結果がまだない時のプレースホルダー */}
        {!loading && results.length === 0 && (
          <div className="text-slate-500 text-center py-20 border-2 border-dashed border-slate-800 rounded-3xl">
            ここに結果が表示されます
          </div>
        )}

        {/* 結果リスト */}
        {results.map((res, index) => (
          <div key={index} className="bg-slate-800/50 p-6 rounded-3xl border border-slate-700">
            <h2 className="text-xl font-bold mb-4 text-slate-300">
              Face #{index + 1}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* 1. 切り抜き結果 */}
              <div className="flex flex-col gap-2">
                <span className="text-cyan-400 font-bold text-sm">✅ 切り抜き結果</span>
                <div className="aspect-square bg-[url('https://media.istockphoto.com/id/1145618475/vector/checkered-flag-pattern.jpg?s=612x612&w=0&k=20&c=A6R_gBwO2Yk1HkU5-qJ5h_yD_T1I_T3W_m_m_m_m')] bg-contain rounded-xl overflow-hidden border-2 border-slate-600 flex items-center justify-center bg-slate-700">
                  <img 
                    src={`data:image/png;base64,${res.base64}`} 
                    className="max-w-full max-h-full object-contain" 
                  />
                </div>
              </div>

              {/* 2. デバッグ画像 (AI視点) */}
              <div className="flex flex-col gap-2">
                <span className="text-red-400 font-bold text-sm">👁 AIの視界 (青:検出 / 赤:探索範囲)</span>
                <div className="aspect-square bg-slate-900 rounded-xl overflow-hidden border-2 border-slate-600 flex items-center justify-center relative">
                  <img 
                    src={`data:image/jpeg;base64,${res.debug_base64}`} 
                    className="max-w-full max-h-full object-contain" 
                  />
                  {/* 凡例 */}
                  <div className="absolute bottom-2 right-2 bg-black/80 p-2 rounded text-xs text-white space-y-1">
                    <div className="flex items-center gap-1"><div className="w-3 h-3 border border-blue-500"></div> 顔検出</div>
                    <div className="flex items-center gap-1"><div className="w-3 h-3 border border-red-500"></div> 探索範囲</div>
                  </div>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  ※ 赤枠の外側は「強制的に削除」されます。<br/>
                  ※ 髪が切れるなら赤枠が髪より小さいです。<br/>
                  ※ 首が残るなら赤枠が下まで伸びすぎています。
                </p>
              </div>

            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
