using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Sentis;
namespace Piper.Scripts
{
    public class PiperManager : MonoBehaviour
    {
        public ModelAsset ModelAsset;
        public int SampleRate = 22050;

        // Piperが必要とする入力スケールなど
        public float ScaleSpeed   = 1.0f;
        public float ScalePitch   = 1.0f;
        public float ScaleGlottal = 0.8f;

        // espeak-ngのdataフォルダ
        public string EspeakNgRelativePath = "espeak-ng-data";
        public string Voice = "en-us";

        Model _runtimeModel;
        Worker _worker;
        
        [SerializeField] BackendType backendType = BackendType.GPUCompute;

        void Awake()
        {
            // 1. PiperWrapperを初期化
            string espeakPath = Path.Combine(Application.streamingAssetsPath, EspeakNgRelativePath);
            PiperWrapper.InitPiper(espeakPath);

            // 2. Sentisモデルを読み込み、Worker作成
            _runtimeModel = ModelLoader.Load(ModelAsset);
            _worker = new Worker(_runtimeModel, backendType);
        }

        /// <summary>
        /// テキストをTTSし、AudioClipを返す非同期メソッド（Taskベース）。
        /// フレームをまたぐ際は Task.Yield() を使い、メインスレッド上でジョブを進めます。
        /// </summary>
        public async Task<AudioClip> TextToSpeechAsync(string text)
        {
            // 3. PiperWrapperでテキストをフォネマイズ
            var phonemeResult = PiperWrapper.ProcessText(text, Voice);
            if (phonemeResult == null || phonemeResult.Sentences == null || phonemeResult.Sentences.Length == 0)
            {
                AppendLog("Phoneme result or sentences are null/empty. Aborting TTS.");
                return null;
            }
            var allSamples = new List<float>();

            // Log model input names, shapes, and types
            AppendLog($"Model expects {(_runtimeModel.inputs?.Count ?? 0)} inputs:");
            if (_runtimeModel.inputs != null)
            {
                for (int i = 0; i < _runtimeModel.inputs.Count; i++)
                {
                    var inp = _runtimeModel.inputs[i];
                    AppendLog($"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.dataType}");
                }
            }

            // 4. 文ごとに推論を実行 & 結合
            for (int s = 0; s < phonemeResult.Sentences.Length; s++)
            {
                var sentence = phonemeResult.Sentences[s];
                // Use object.ReferenceEquals for null check, since 'PiperProcessedSentence' does not support '==' or '?'
                if (object.ReferenceEquals(sentence, null) || sentence.PhonemesIds == null || sentence.PhonemesIds.Length == 0)
                {
                    AppendLog($"Sentence {s} or its phoneme IDs are null/empty. Skipping.");
                    continue;
                }
                int[] phonemeIds = sentence.PhonemesIds;

                // 入力テンソル作成
                using var inputTensor = new Tensor<int>(new TensorShape(1, phonemeIds.Length), phonemeIds);
                using var inputLengthsTensor = new Tensor<int>(new TensorShape(1), new int[] { phonemeIds.Length });
                using var scalesTensor = new Tensor<float>(
                    new TensorShape(3),
                    new float[] { ScaleSpeed, ScalePitch, ScaleGlottal }
                );

                // 入力名をモデルに合わせる (たとえば 0=input, 1=input_lengths, 2=scales)
                if (_runtimeModel.inputs == null || _runtimeModel.inputs.Count < 3)
                {
                    AppendLog("Model does not have enough inputs defined. Aborting.");
                    return null;
                }
                string inputName        = _runtimeModel.inputs[0].name;
                string inputLengthsName = _runtimeModel.inputs[1].name;
                string scalesName       = _runtimeModel.inputs[2].name;

                if (inputTensor == null)
                {
                    AppendLog($"inputTensor is null for input: {inputName}. Skipping this sentence.");
                    continue;
                }
                if (inputLengthsTensor == null)
                {
                    AppendLog($"inputLengthsTensor is null for input: {inputLengthsName}. Skipping this sentence.");
                    continue;
                }
                if (scalesTensor == null)
                {
                    AppendLog($"scalesTensor is null for input: {scalesName}. Skipping this sentence.");
                    continue;
                }

                // Debug: Log tensor values and shapes before inference
                AppendLog($"inputTensor shape: {inputTensor.shape}, values: [{string.Join(",", phonemeIds)}]");
                AppendLog($"inputLengthsTensor shape: {inputLengthsTensor.shape}, values: [{phonemeIds.Length}]");
                AppendLog($"scalesTensor shape: {scalesTensor.shape}, values: [{ScaleSpeed},{ScalePitch},{ScaleGlottal}]");
                AppendLog($"Setting input: {inputName}, shape: {inputTensor.shape}, type: {inputTensor.dataType}");
                AppendLog($"Setting input: {inputLengthsName}, shape: {inputLengthsTensor.shape}, type: {inputLengthsTensor.dataType}");
                AppendLog($"Setting input: {scalesName}, shape: {scalesTensor.shape}, type: {scalesTensor.dataType}");

                _worker.SetInput(inputName,         inputTensor);
                _worker.SetInput(inputLengthsName,  inputLengthsTensor);
                _worker.SetInput(scalesName,        scalesTensor);

                // スケジュール実行
                _worker.Schedule();

                // 4-1. ScheduleIterableでジョブをフレームまたぎ進行
                var enumerator = _worker.ScheduleIterable();
                while (enumerator.MoveNext())
                {
                    // コルーチンの代わりに Task.Yield() で1フレーム中断
                    await Task.Yield();
                }

                // 4-2. 出力を取得
                var outputRaw = _worker.PeekOutput();
                if (outputRaw == null)
                {
                    AppendLog("Output tensor is null. Skipping this sentence.");
                    continue;
                }
                AppendLog($"Output type: {outputRaw.GetType().Name}");
                if (outputRaw is Tensor<float> outputTensor)
                {
                    AppendLog($"Output tensor shape: {outputTensor.shape}");
                    float[] sentenceSamples = outputTensor.DownloadToArray();
                    allSamples.AddRange(sentenceSamples);
                }
                else
                {
                    AppendLog($"Output is not a Tensor<float>, but {outputRaw.GetType().Name}. Skipping this sentence.");
                    continue;
                }
            }

            // 5. 音声波形をまとめて AudioClip 作成
            if (allSamples.Count == 0)
            {
                AppendLog("No audio samples generated. Returning null AudioClip.");
                return null;
            }
            AudioClip clip = AudioClip.Create("PiperTTS", allSamples.Count, 1, SampleRate, false);
            clip.SetData(allSamples.ToArray(), 0);

            return clip;
        }

        private string _logBuffer = "";
        private Vector2 _logScroll;

        private void AppendLog(string message)
        {
            Debug.Log(message);
            _logBuffer += message + "\n";
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, Screen.width - 20, Screen.height / 3));
            GUILayout.Label("PiperManager Debug Log (copy and paste below):");
            _logScroll = GUILayout.BeginScrollView(_logScroll, GUILayout.Height(Screen.height / 3 - 30));
            GUI.SetNextControlName("LogTextArea");
            _logBuffer = GUILayout.TextArea(_logBuffer, GUILayout.ExpandHeight(true));
            GUILayout.EndScrollView();
            if (GUILayout.Button("Copy to Clipboard"))
            {
                GUI.FocusControl("LogTextArea");
                TextEditor te = new TextEditor { text = _logBuffer };
                te.SelectAll();
                te.Copy();
            }
            GUILayout.EndArea();
        }

        void OnDestroy()
        {
            PiperWrapper.FreePiper();
            if (_worker != null)
            {
                _worker.Dispose();
            }
        }
    }
}
