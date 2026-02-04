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
  color_correction_strength: number;  // 使用された色補正強度
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
  const [colorCorrection, setColorCorrection] = useState<number | null>(null); // 色補正強度 (0-1)、nullの場合は自動

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
        targetPath,
        colorCorrection: colorCorrection !== null ? colorCorrection : undefined
      });
      setSwapResult(result.base64);
      // 自動計算された色補正強度をスライダーに反映
      setColorCorrection(result.color_correction_strength);
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white flex flex-col items-center p-4 sm:p-8">
      {/* ヘッダー */}
      <div className="text-center mb-8">
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-3 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
          Face Swapper AI
        </h1>
        <p className="text-slate-400 text-sm sm:text-base">高精度な顔入れ替えツール - Powered by OpenCV</p>
      </div>

      {/* モード切り替え */}
      <div className="mb-8 flex gap-2 bg-slate-800/70 backdrop-blur-sm p-1.5 rounded-full shadow-lg border border-slate-700">
        <button 
          onClick={() => setMode('swap')}
          className={`px-5 sm:px-7 py-2.5 rounded-full font-bold transition-all duration-300 text-sm sm:text-base ${
            mode === 'swap' 
              ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-lg shadow-cyan-500/50 scale-105' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          🔄 Face Swap
        </button>
        <button 
          onClick={() => setMode('extract')}
          className={`px-5 sm:px-7 py-2.5 rounded-full font-bold transition-all duration-300 text-sm sm:text-base ${
            mode === 'extract' 
              ? 'bg-gradient-to-r from-purple-500 to-pink-500 shadow-lg shadow-purple-500/50 scale-105' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          ✂️ Face Extract
        </button>
      </div>

      {/* Face Swap Mode */}
      {mode === 'swap' && (
        <div className="w-full max-w-6xl space-y-6">
          {/* 画像選択エリア */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            {/* Source Image */}
            <div className="bg-gradient-to-br from-slate-800/70 to-cyan-900/20 backdrop-blur-sm p-5 sm:p-6 rounded-2xl border-2 border-cyan-500/30 hover:border-cyan-500/60 transition-all duration-300 shadow-xl">
              <h3 className="text-lg sm:text-xl font-bold mb-3 text-cyan-400 flex items-center gap-2">
                <span className="text-2xl">①</span> この顔を使う
              </h3>
              <div 
                onClick={selectSourceImage}
                className="aspect-square bg-slate-900/70 rounded-xl border-2 border-dashed border-slate-600 hover:border-cyan-400 hover:bg-slate-900/90 cursor-pointer flex items-center justify-center transition-all duration-300 overflow-hidden group hover:scale-[1.02]"
              >
                {sourcePreview ? (
                  <img src={`asset://localhost/${sourcePreview}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                ) : (
                  <div className="text-center text-slate-500 group-hover:text-cyan-400 transition-colors duration-300">
                    <div className="text-5xl mb-2 group-hover:scale-110 transition-transform">📷</div>
                    <div className="text-sm sm:text-base">クリックして画像を選択</div>
                  </div>
                )}
              </div>
              {sourcePath && (
                <div className="mt-3 px-3 py-2 bg-slate-900/50 rounded-lg">
                  <p className="text-xs text-slate-400 truncate">📁 {sourcePath.split('/').pop()}</p>
                </div>
              )}
            </div>

            {/* Target Image */}
            <div className="bg-gradient-to-br from-slate-800/70 to-purple-900/20 backdrop-blur-sm p-5 sm:p-6 rounded-2xl border-2 border-purple-500/30 hover:border-purple-500/60 transition-all duration-300 shadow-xl">
              <h3 className="text-lg sm:text-xl font-bold mb-3 text-purple-400 flex items-center gap-2">
                <span className="text-2xl">②</span> この画像に埋め込む
              </h3>
              <div 
                onClick={selectTargetImage}
                className="aspect-square bg-slate-900/70 rounded-xl border-2 border-dashed border-slate-600 hover:border-purple-400 hover:bg-slate-900/90 cursor-pointer flex items-center justify-center transition-all duration-300 overflow-hidden group hover:scale-[1.02]"
              >
                {targetPreview ? (
                  <img src={`asset://localhost/${targetPreview}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                ) : (
                  <div className="text-center text-slate-500 group-hover:text-purple-400 transition-colors duration-300">
                    <div className="text-5xl mb-2 group-hover:scale-110 transition-transform">🖼️</div>
                    <div className="text-sm sm:text-base">クリックして画像を選択</div>
                  </div>
                )}
              </div>
              {targetPath && (
                <div className="mt-3 px-3 py-2 bg-slate-900/50 rounded-lg">
                  <p className="text-xs text-slate-400 truncate">📁 {targetPath.split('/').pop()}</p>
                </div>
              )}
            </div>
          </div>

          {/* 色補正スライダー */}
          <div className="bg-gradient-to-br from-slate-800/70 to-slate-900/70 backdrop-blur-sm p-5 sm:p-6 rounded-2xl border border-slate-700/50 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <label className="text-sm sm:text-base font-bold text-slate-200 flex items-center gap-2">
                <span className="text-xl">🎨</span> 色補正の強度
              </label>
              <span className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded-full text-cyan-400 font-mono text-sm font-bold">
                {colorCorrection !== null ? Math.round(colorCorrection * 100) + '%' : '自動'}
              </span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0"
                max="100"
                value={colorCorrection !== null ? colorCorrection * 100 : 50}
                onChange={(e) => setColorCorrection(Number(e.target.value) / 100)}
                className="w-full h-3 bg-slate-700/50 rounded-full appearance-none cursor-pointer accent-cyan-500 
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 
                  [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r 
                  [&::-webkit-slider-thumb]:from-cyan-400 [&::-webkit-slider-thumb]:to-blue-500 
                  [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-cyan-500/50
                  [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:hover:scale-110
                  [&::-webkit-slider-thumb]:transition-transform"
              />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-3 px-1">
              <span>💡 弱（元の色を保持）</span>
              <span>🔥 強（完全に合わせる）</span>
            </div>
          </div>

          {/* Swap Button */}
          <div className="flex justify-center py-4">
            <button
              onClick={performFaceSwap}
              disabled={swapping || !sourcePath || !targetPath}
              className="group relative px-10 sm:px-14 py-4 sm:py-5 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 
                hover:shadow-2xl hover:shadow-purple-500/50 hover:scale-105
                disabled:from-slate-700 disabled:to-slate-700 disabled:shadow-none disabled:scale-100
                rounded-full font-bold text-lg sm:text-xl transition-all duration-300 
                flex items-center gap-3 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 opacity-0 group-hover:opacity-20 transition-opacity" />
              {swapping ? (
                <>
                  <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                  <span className="relative">処理中...</span>
                </>
              ) : (
                <>
                  <span className="text-2xl group-hover:rotate-12 transition-transform">✨</span>
                  <span className="relative">Face Swap を実行</span>
                  <span className="text-2xl group-hover:-rotate-12 transition-transform">✨</span>
                </>
              )}
            </button>
          </div>

          {/* Result */}
          {swapResult && (
            <div className="bg-gradient-to-br from-slate-800/90 to-purple-900/40 backdrop-blur-lg p-6 sm:p-8 rounded-2xl border-2 border-purple-500/50 shadow-2xl shadow-purple-500/20 animate-in fade-in duration-500">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-5">
                <h3 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-2">
                  <span className="text-3xl animate-bounce">🎉</span> 完成！
                </h3>
                <button
                  onClick={downloadResult}
                  className="px-5 sm:px-6 py-2.5 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 
                    rounded-full font-bold transition-all duration-300 flex items-center gap-2 shadow-lg hover:shadow-green-500/50 hover:scale-105"
                >
                  <span className="text-lg">💾</span> ダウンロード
                </button>
              </div>
              <div className="bg-slate-900/60 backdrop-blur rounded-xl overflow-hidden border-2 border-slate-700/50 flex items-center justify-center p-4 hover:border-purple-500/50 transition-colors">
                <img 
                  src={`data:image/png;base64,${swapResult}`} 
                  className="max-w-full max-h-[600px] object-contain rounded-lg shadow-2xl" 
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
