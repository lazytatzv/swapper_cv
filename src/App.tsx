import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from '@tauri-apps/plugin-dialog';
import "./App.css";

// Rustから返ってくるデータの型定義
interface FaceResult {
  base64: string;       // 切り抜き後の画像 (PNG)
  debug_base64: string; // 解析用画像 (JPG: 赤枠・青枠付き)
}

interface FaceSwapResult {
  base64: string;  // 合成結果画像
}

function App() {
  const [mode, setMode] = useState<'extract' | 'swap'>('swap');
  
  // Face Extraction用
  const [results, setResults] = useState<FaceResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  // Face Swap用
  const [sourcePath, setSourcePath] = useState<string>("");
  const [targetPath, setTargetPath] = useState<string>("");
  const [sourcePreview, setSourcePreview] = useState<string>("");
  const [targetPreview, setTargetPreview] = useState<string>("");
  const [swapResult, setSwapResult] = useState<string>("");
  const [swapping, setSwapping] = useState<boolean>(false);

  const selectAndProcess = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
    });

    if (file && typeof file === 'string') {
      setLoading(true);
      setResults([]);
      try {
        const data = await invoke<FaceResult[]>("process_face", { path: file });
        setResults(data);
      } catch (e) {
        console.error(e);
        alert("エラーが発生しました: " + e);
      } finally {
        setLoading(false);
      }
    }
  };

  const selectSourceImage = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
    });

    if (file && typeof file === 'string') {
      setSourcePath(file);
      // プレビュー用にファイルパスをそのまま使用（Tauriの場合convertFileSrcを使う方が良い）
      setSourcePreview(file);
    }
  };

  const selectTargetImage = async () => {
    const file = await open({
      multiple: false,
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
    });

    if (file && typeof file === 'string') {
      setTargetPath(file);
      setTargetPreview(file);
    }
  };

  const performFaceSwap = async () => {
    if (!sourcePath || !targetPath) {
      alert("両方の画像を選択してください");
      return;
    }

    setSwapping(true);
    setSwapResult("");
    try {
      const result = await invoke<FaceSwapResult>("face_swap", { 
        sourcePath, 
        targetPath 
      });
      setSwapResult(result.base64);
    } catch (e) {
      console.error(e);
      alert("Face Swapに失敗しました: " + e);
    } finally {
      setSwapping(false);
    }
  };

  const downloadResult = () => {
    if (!swapResult) return;
    
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${swapResult}`;
    link.download = `faceswap_${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex flex-col items-center p-8">
      <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
        Face Swapper AI
      </h1>
      <p className="text-slate-400 mb-8">高精度な顔入れ替えツール</p>

      {/* モード切り替え */}
      <div className="mb-8 flex gap-2 bg-slate-800/50 p-1 rounded-full">
        <button 
          onClick={() => setMode('swap')}
          className={`px-6 py-2 rounded-full font-bold transition-all ${
            mode === 'swap' 
              ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-lg' 
              : 'text-slate-400 hover:text-white'
          }`}
        >
          🔄 Face Swap
        </button>
        <button 
          onClick={() => setMode('extract')}
          className={`px-6 py-2 rounded-full font-bold transition-all ${
            mode === 'extract' 
              ? 'bg-gradient-to-r from-purple-500 to-pink-500 shadow-lg' 
              : 'text-slate-400 hover:text-white'
          }`}
        >
          ✂️ Face Extract
        </button>
      </div>

      {/* Face Swap Mode */}
      {mode === 'swap' && (
        <div className="w-full max-w-6xl space-y-8">
          {/* 画像選択エリア */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Source Image */}
            <div className="bg-slate-800/50 p-6 rounded-3xl border-2 border-cyan-500/30">
              <h3 className="text-xl font-bold mb-4 text-cyan-400">① この顔を使う</h3>
              <div 
                onClick={selectSourceImage}
                className="aspect-square bg-slate-900/50 rounded-xl border-2 border-dashed border-slate-600 hover:border-cyan-500 cursor-pointer flex items-center justify-center transition-all overflow-hidden group"
              >
                {sourcePreview ? (
                  <img src={`asset://localhost/${sourcePreview}`} className="w-full h-full object-cover" />
                ) : (
                  <div className="text-center text-slate-500 group-hover:text-cyan-400 transition-colors">
                    <div className="text-5xl mb-2">📷</div>
                    <div>クリックして画像を選択</div>
                  </div>
                )}
              </div>
              {sourcePath && (
                <div className="mt-2 text-xs text-slate-400 truncate">
                  {sourcePath.split('/').pop()}
                </div>
              )}
            </div>

            {/* Target Image */}
            <div className="bg-slate-800/50 p-6 rounded-3xl border-2 border-purple-500/30">
              <h3 className="text-xl font-bold mb-4 text-purple-400">② この画像に埋め込む</h3>
              <div 
                onClick={selectTargetImage}
                className="aspect-square bg-slate-900/50 rounded-xl border-2 border-dashed border-slate-600 hover:border-purple-500 cursor-pointer flex items-center justify-center transition-all overflow-hidden group"
              >
                {targetPreview ? (
                  <img src={`asset://localhost/${targetPreview}`} className="w-full h-full object-cover" />
                ) : (
                  <div className="text-center text-slate-500 group-hover:text-purple-400 transition-colors">
                    <div className="text-5xl mb-2">🖼️</div>
                    <div>クリックして画像を選択</div>
                  </div>
                )}
              </div>
              {targetPath && (
                <div className="mt-2 text-xs text-slate-400 truncate">
                  {targetPath.split('/').pop()}
                </div>
              )}
            </div>
          </div>

          {/* Swap Button */}
          <div className="flex justify-center">
            <button
              onClick={performFaceSwap}
              disabled={swapping || !sourcePath || !targetPath}
              className="px-12 py-4 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 hover:shadow-2xl hover:shadow-purple-500/50 disabled:from-slate-700 disabled:to-slate-700 disabled:shadow-none rounded-full font-bold text-xl transition-all flex items-center gap-3"
            >
              {swapping ? (
                <>
                  <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  処理中...
                </>
              ) : (
                <>
                  <span>✨</span>
                  Face Swap を実行
                  <span>✨</span>
                </>
              )}
            </button>
          </div>

          {/* Result */}
          {swapResult && (
            <div className="bg-gradient-to-br from-slate-800/80 to-purple-900/30 p-8 rounded-3xl border-2 border-purple-500/50 backdrop-blur">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-pink-400 bg-clip-text text-transparent">
                  🎉 完成！
                </h3>
                <button
                  onClick={downloadResult}
                  className="px-6 py-2 bg-green-600 hover:bg-green-500 rounded-full font-bold transition-all flex items-center gap-2"
                >
                  💾 ダウンロード
                </button>
              </div>
              <div className="bg-slate-900/50 rounded-xl overflow-hidden border-2 border-slate-700 flex items-center justify-center">
                <img 
                  src={`data:image/png;base64,${swapResult}`} 
                  className="max-w-full max-h-[600px] object-contain" 
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Face Extract Mode */}
      {mode === 'extract' && (
        <div className="w-full max-w-6xl space-y-8">
          <div className="flex justify-center">
            <button 
              onClick={selectAndProcess}
              disabled={loading}
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:shadow-2xl hover:shadow-purple-500/50 disabled:from-slate-700 disabled:to-slate-700 rounded-full font-bold text-lg transition-all shadow-lg flex items-center gap-2"
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

          {!loading && results.length === 0 && (
            <div className="text-slate-500 text-center py-20 border-2 border-dashed border-slate-800 rounded-3xl">
              ここに結果が表示されます
            </div>
          )}

          {results.map((res, index) => (
            <div key={index} className="bg-slate-800/50 p-6 rounded-3xl border border-slate-700">
              <h2 className="text-xl font-bold mb-4 text-slate-300">
                Face #{index + 1}
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex flex-col gap-2">
                  <span className="text-cyan-400 font-bold text-sm">✅ 切り抜き結果</span>
                  <div className="aspect-square bg-slate-700 rounded-xl overflow-hidden border-2 border-slate-600 flex items-center justify-center">
                    <img 
                      src={`data:image/png;base64,${res.base64}`} 
                      className="max-w-full max-h-full object-contain" 
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <span className="text-red-400 font-bold text-sm">👁 AIの視界 (青:検出 / 赤:探索範囲)</span>
                  <div className="aspect-square bg-slate-900 rounded-xl overflow-hidden border-2 border-slate-600 flex items-center justify-center relative">
                    <img 
                      src={`data:image/jpeg;base64,${res.debug_base64}`} 
                      className="max-w-full max-h-full object-contain" 
                    />
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
      )}
    </div>
  );
}

export default App;
