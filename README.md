# Real-time Music Genre Classification

โปรเจคนี้เป็นระบบจำแนกแนวเพลงแบบ real time โดยใช้โมเดล deep learning ที่ train กับ audio chunk ยาว 15 วินาที แล้วนำมาใช้งานได้ทั้งจากไฟล์เพลงและไมโครโฟน

## แอปนี้ทำอะไรได้บ้าง

- เลือก model จาก folder `models/` ผ่านเมนู ไม่ต้องพิมพ์ path เอง
- เลือกเพลงจาก folder `songs/` ผ่านเมนู รองรับไฟล์เช่น `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`
- เลือกได้ว่าจะใช้ input จากไฟล์เพลงหรือจาก microphone
- ประมวลผลแบบ sliding window:
  - model window = `15 seconds`
  - hop = `1 second`
  - แสดงผลใหม่ทุก 1 วินาทีหลังจากมีเสียงครบ 15 วินาทีแรก
- แสดง realtime prediction ด้วย soft vote จาก probability ล่าสุด 3 windows
- ตอนจบสรุป final genre ด้วย majority vote
- รองรับ model หลายรูปแบบ:
  - `.pt`, `.pth`, `.ckpt` สำหรับ PyTorch checkpoint
  - `.onnx` สำหรับ ONNX Runtime
- รองรับ ensemble หลายโมเดลด้วยการส่ง `--model` หลายครั้ง

## สิ่งที่ทำไปแล้วใน realtime pipeline

ไฟล์หลักคือ `realtime.py`

สิ่งที่ปรับจาก pipeline เดิม:

- เปลี่ยนจาก chunk สั้นแล้ว pad เป็น 30 วินาที เป็น sliding window 15 วินาทีตามที่โมเดลใหม่ train มา
- เพิ่ม hop 1 วินาที เพื่อให้ระบบแสดงผลแบบ realtime ได้ถี่ขึ้น
- เพิ่ม soft vote 3 windows สำหรับผลที่โชว์ระหว่างเพลง
- เพิ่ม majority vote สำหรับผลสรุปสุดท้าย
- เพิ่ม loader สำหรับ `.pt` checkpoint ตัวใหม่ `cnn_specaug_best.pt`
- อ่าน config จาก checkpoint อัตโนมัติ เช่น sample rate, n_mels, n_fft, fmin, fmax, genre list
- เพิ่ม interactive menu เพื่อให้ใช้งานง่ายขึ้น ไม่ต้องพิมพ์ path ยาว ๆ
- ยังรองรับ command-line arguments สำหรับคนที่อยากทดลองละเอียดหรือเอาไปต่อยอด

## โครงสร้าง folder ที่แนะนำ

```text
pattern-music/
├── realtime.py
├── README.md
├── models/
│   └── cnn_specaug_best.pt
├── songs/
│   └── your_song.mp3
├── Data/
│   └── ...
└── src/
    └── ...
```

ใส่ model ที่ต้องการใช้ไว้ใน `models/`

ใส่เพลงที่ต้องการทดสอบไว้ใน `songs/`

## วิธีใช้งานแบบง่ายที่สุด

เปิด terminal ที่ root ของโปรเจค:

```powershell
cd C:\Y3T2\PATTERN\pattern-music
python realtime.py
```

จากนั้นโปรแกรมจะขึ้นเมนูให้เลือก:

1. model จาก folder `models/`
2. input source ว่าจะใช้ไฟล์เพลงหรือ microphone
3. ถ้าเลือกไฟล์เพลง โปรแกรมจะให้เลือกเพลงจาก folder `songs/`

## ใช้งานกับไฟล์เพลง

วางไฟล์เพลงไว้ใน `songs/` ก่อน เช่น:

```text
songs/my_song.mp3
```

แล้วรัน:

```powershell
python realtime.py
```

หรือระบุ mode ให้เข้าเมนูเลือกเพลงทันที:

```powershell
python realtime.py --mode file
```

ถ้าต้องการระบุ path เอง:

```powershell
python realtime.py --model "models\cnn_specaug_best.pt" --file "songs\my_song.mp3" --simulate-realtime
```

`--simulate-realtime` ทำให้การอ่านไฟล์เพลงรอทีละ 1 วินาทีเหมือน realtime จริง เหมาะสำหรับ demo

## ใช้งานกับ microphone

รัน:

```powershell
python realtime.py --mode mic
```

หรือเข้าเมนูปกติ:

```powershell
python realtime.py
```

แล้วเลือก `Microphone`

หมายเหตุ: microphone mode ต้องรอเสียงครบ 15 วินาทีแรกก่อน ถึงจะเริ่มแสดง prediction จากนั้นจะ update ทุก 1 วินาที

## Model ที่ใช้อยู่

โมเดลหลักตอนนี้คือ:

```text
models/cnn_specaug_best.pt
```

checkpoint นี้มี config ฝังอยู่ในไฟล์ เช่น:

- sample rate: `22050`
- chunk duration: `15.0 seconds`
- n_mels: `128`
- n_fft: `2048`
- mel hop length: `512`
- fmin: `20`
- fmax: `8000`
- number of classes: `8`

genre classes:

```text
Electronic
Experimental
Folk
Hip-Hop
Instrumental
International
Pop
Rock
```

## อธิบาย realtime.py แบบละเอียด

section นี้อธิบาย flow ของไฟล์ `realtime.py` เพื่อให้คนในกลุ่มเข้าใจว่าโปรแกรมทำงานยังไง และจะแก้หรือต่อยอดตรงไหนได้บ้าง

### ภาพรวมการทำงาน

เวลาเรา run:

```powershell
python realtime.py
```

โปรแกรมจะทำงานตามลำดับนี้:

1. เปิด interactive menu
2. หา model ใน folder `models/`
3. ให้เลือกว่าจะใช้ไฟล์เพลงหรือ microphone
4. ถ้าเลือกไฟล์เพลง จะหาเพลงใน folder `songs/`
5. โหลด model และ config ของ model
6. สร้าง audio preprocessor ให้ตรงกับตอน train
7. อ่านเสียงเข้ามาเป็น waveform
8. ตัดเสียงเป็น sliding window ยาว 15 วินาที
9. เลื่อน window ทุก 1 วินาที
10. แปลง waveform เป็น mel-spectrogram
11. ส่งเข้า model เพื่อ predict probability ของแต่ละ genre
12. ใช้ soft vote จาก 3 windows ล่าสุดเพื่อแสดงผล realtime
13. เมื่อจบเพลงหรือกด `Ctrl+C` จะสรุป final genre ด้วย majority vote

flow สั้น ๆ:

```text
audio input
-> 15s sliding window
-> mel-spectrogram
-> model inference
-> probability per genre
-> 3-window soft vote for realtime display
-> majority vote for final result
```

### ส่วน config ด้านบนของไฟล์

ค่าหลักที่อยู่ด้านบนของ `realtime.py`:

```python
SAMPLE_RATE = 22050
WINDOW_SEC = 15.0
HOP_SEC = 1.0
VOTE_WINDOWS = 3

N_MELS = 128
N_FFT = 2048
MEL_HOP_LENGTH = 512
F_MIN = 0.0
F_MAX = None
```

ความหมาย:

- `SAMPLE_RATE` คือ sample rate ที่ใช้ประมวลผลเสียง
- `WINDOW_SEC` คือความยาวเสียงที่ model เห็นต่อการ predict 1 ครั้ง
- `HOP_SEC` คือระยะที่เลื่อน window ต่อครั้ง
- `VOTE_WINDOWS` คือจำนวน windows ล่าสุดที่เอามาเฉลี่ย probability เพื่อโชว์ realtime
- `N_MELS`, `N_FFT`, `MEL_HOP_LENGTH`, `F_MIN`, `F_MAX` คือค่าที่ใช้สร้าง mel-spectrogram

สำหรับ `cnn_specaug_best.pt` โปรแกรมจะอ่าน config จาก checkpoint อัตโนมัติ ทำให้ไม่ต้องแก้ค่าเหล่านี้เองถ้า model มี config ฝังอยู่

### Interactive menu

ส่วนนี้ทำให้ไม่ต้องพิมพ์ path เอง:

```python
apply_interactive_selection(args)
```

หน้าที่หลัก:

- ใช้ `models/` เป็น folder สำหรับเลือก model
- ใช้ `songs/` เป็น folder สำหรับเลือกเพลง
- แสดง list ของไฟล์ model ที่เจอ เช่น `.pt`, `.pth`, `.ckpt`, `.onnx`
- แสดง list ของไฟล์เพลงที่เจอ เช่น `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`
- ถ้าเลือก file mode จะตั้ง `--simulate-realtime` ให้อัตโนมัติ เพื่อให้ demo เหมือน realtime จริง

function ที่เกี่ยวข้อง:

```python
find_files(folder, extensions)
prompt_choice(title, items, base)
prompt_mode()
apply_interactive_selection(args)
```

ถ้าต้องการเปลี่ยน folder เริ่มต้น สามารถใช้:

```powershell
python realtime.py --models-dir "my_models" --songs-dir "my_songs"
```

### การโหลด model

โปรแกรมรองรับทั้ง PyTorch และ ONNX ผ่าน backend คนละแบบ

function สำคัญ:

```python
load_backends(...)
load_torch_backend(...)
load_onnx_backend(...)
```

ถ้า model เป็น `.onnx`:

```text
load_onnx_backend
-> onnxruntime.InferenceSession
-> run inference ด้วย session.run(...)
```

ถ้า model เป็น `.pt`, `.pth`, `.ckpt`:

```text
load_torch_backend
-> ลองโหลดเป็น TorchScript ก่อน
-> ถ้าไม่ใช่ TorchScript จะโหลด checkpoint
-> หา model_state_dict
-> สร้าง architecture
-> load_state_dict
-> model.eval()
```

สำหรับ `cnn_specaug_best.pt` โปรแกรมสร้าง architecture ให้เองด้วย class:

```python
SpecAugCnn
```

และตรวจว่า checkpoint ใช่ architecture นี้ไหมด้วย key เช่น:

```text
features.0.block.0.weight
features.2.block.0.weight
features.4.block.0.weight
features.6.block.0.weight
classifier.1.weight
classifier.4.weight
```

ถ้าเจอ key เหล่านี้ โปรแกรมจะรู้ว่าเป็น CNN ตัวนี้ แล้วสร้าง model ให้เองโดยไม่ต้องมีไฟล์ model definition เพิ่ม

สำหรับ `pond_best.pt` checkpoint นี้เป็น `state_dict` ของ MobileNetV2-like model ไม่ใช่ full model ดังนั้นโปรแกรมต้อง rebuild architecture ก่อนแล้วค่อย `load_state_dict` ตอนนี้ `realtime.py` รองรับตัวนี้แล้วผ่าน `torchvision.models.mobilenet_v2` โดยปรับ first convolution ให้รับ mel 1 channel:

```text
features.0.0.weight = (32, 1, 3, 3)
classifier.1.weight = (8, 1280)
```

ถ้าไม่มี fallback นี้จะเจอ error:

```text
contains weights only. Pass --model-factory ...
```

สำหรับ `ast_fma_best.pt` checkpoint นี้เป็น `state_dict` ของ HuggingFace `ASTForAudioClassification` โปรแกรมจะ rebuild architecture ด้วย `transformers` แล้วแปลง input จาก mel format ของ pipeline ให้เป็น format ที่ AST ต้องการ:

```text
pipeline mel: (batch, 1, 128, T)
AST input:    (batch, 1024, 128)
```

AST ต้องใช้ `transformers`:

```powershell
pip install transformers
```

หมายเหตุ: AST เป็น transformer model ขนาดใหญ่กว่า CNN/EfficientNet/MobileNet มาก บน CPU อาจใช้เวลาประมาณ 1.7-2.0 วินาทีต่อ 1 prediction ซึ่งช้ากว่า hop 1 วินาที ถ้าต้องการใช้ realtime จาก microphone จริง ๆ แนะนำให้ใช้ GPU หรือเลือก model ที่เบากว่า เช่น `cnn_specaug_best.pt`, `efficientnet_b0_full_model.pt`, หรือ `pond_best.pt`

สำหรับ model กลุ่ม EfficientNet เช่น `efficientnet_b0_full_model.pt` โมเดลเป็น image model จาก `timm` ที่ต้องการ input 3 channels แต่ mel-spectrogram ของ pipeline ปกติเป็น 1 channel โปรแกรมจึงตรวจ first conv ของโมเดล แล้วถ้าเจอว่าโมเดลต้องการ 3 channels จะ duplicate mel channel ให้อัตโนมัติ:

```text
(1, 1, 128, T)
-> repeat channel
-> (1, 3, 128, T)
```

ถ้าไม่ทำขั้นตอนนี้จะเจอ error ประมาณ:

```text
expected input to have 3 channels, but got 1 channels instead
```

### Architecture ของ cnn_specaug_best.pt

architecture ที่ reconstruct ใน `realtime.py` คือ:

```text
Input mel-spectrogram
-> Conv2d 1 to 32 + BatchNorm + ReLU
-> MaxPool2d
-> Conv2d 32 to 64 + BatchNorm + ReLU
-> MaxPool2d
-> Conv2d 64 to 128 + BatchNorm + ReLU
-> MaxPool2d
-> Conv2d 128 to 256 + BatchNorm + ReLU
-> AdaptiveAvgPool2d
-> Flatten
-> Linear 256 to 256
-> ReLU
-> Dropout
-> Linear 256 to 8 genres
```

class ที่เกี่ยวข้อง:

```python
ConvBnRelu
SpecAugCnn
build_builtin_model(...)
```

ถ้าในอนาคตใช้ checkpoint architecture ใหม่ที่ key ไม่เหมือนตัวนี้ ต้องเพิ่ม class model ใหม่ หรือใช้ `--model-factory`

ตัวอย่าง `--model-factory`:

```powershell
python realtime.py --model "models\new_model.pt" --model-factory "src\model_def.py:create_model"
```

โดย `create_model()` ต้อง return `torch.nn.Module`

### การอ่าน config จาก checkpoint

`cnn_specaug_best.pt` มี field ชื่อ `config` อยู่ใน checkpoint

โปรแกรมอ่านด้วย:

```python
extract_config(checkpoint)
extract_class_names(checkpoint)
```

ค่าที่อ่านมาใช้ เช่น:

```text
sample_rate
chunk_duration
n_mels
n_fft
hop_length
fmin
fmax
genres
num_classes
cnn_dropout
```

ข้อดีคือถ้าเพื่อนย้าย model ไปเครื่องอื่น โปรแกรมยังรู้ว่า model นี้ train ด้วย setting อะไร ไม่ต้องจำเองทั้งหมด

### AudioPreprocessor

class นี้รับ waveform แล้วแปลงเป็น input ที่ model ต้องการ:

```python
AudioPreprocessor
```

ขั้นตอนใน preprocessor:

1. รับ waveform เป็น numpy array
2. ถ้าเสียงสั้นกว่า 15 วินาที จะ pad ด้วย 0
3. ถ้าเสียงยาวกว่า 15 วินาที จะเอาเฉพาะ 15 วินาทีล่าสุด
4. แปลงเป็น torch tensor
5. สร้าง mel-spectrogram
6. แปลง amplitude เป็น dB
7. normalize ด้วย mean/std ของ mel นั้น
8. เพิ่ม dimension ให้เป็น shape ที่ model ต้องการ

input format default คือ:

```text
mel4d
```

shape ที่ส่งเข้า model:

```text
(batch, channel, n_mels, time)
```

หรือประมาณ:

```text
(1, 1, 128, T)
```

### File mode ทำงานยังไง

ถ้าเลือกเพลงจาก `songs/` หรือส่ง `--file` โปรแกรมจะเข้า function:

```python
run_from_file(...)
```

ขั้นตอน:

1. โหลดไฟล์เสียงด้วย `load_audio_file(...)`
2. ถ้าเป็น stereo จะรวมเป็น mono
3. ถ้า sample rate ไม่ตรง จะ resample เป็น sample rate ที่ model ต้องการ
4. สร้าง sliding windows ด้วย `iter_file_windows(...)`
5. predict ทีละ window
6. แสดงผล realtime
7. สรุป final result ตอนจบ

การตัด window:

```text
window 1: 00s ถึง 15s
window 2: 01s ถึง 16s
window 3: 02s ถึง 17s
...
```

เพราะ:

```text
window = 15s
hop = 1s
```

ถ้าเปิด `--simulate-realtime` โปรแกรมจะ sleep ให้แต่ละรอบห่างกันประมาณ 1 วินาที เหมือนเพลงกำลังเล่นจริง

### Microphone mode ทำงานยังไง

ถ้าเลือก microphone โปรแกรมจะเข้า function:

```python
run_realtime(...)
```

ขั้นตอน:

1. เปิด microphone stream ด้วย `sounddevice.InputStream`
2. callback รับเสียงเข้ามาทีละ block
3. เก็บเสียงไว้ใน `queue`
4. สะสมเสียงใน `buffer`
5. รอจน buffer มีเสียงครบ 15 วินาที
6. หลังจากนั้นทุก ๆ 1 วินาทีจะ predict 1 ครั้ง
7. แต่ละ prediction ใช้เสียงย้อนหลัง 15 วินาทีล่าสุด

แนวคิดของ buffer:

```text
เก็บเสียงล่าสุดไว้ไม่เกิน 15 วินาที
ทุก 1 วินาที เอา buffer นี้ไป predict
```

ดังนั้น microphone mode จะไม่แสดงผลทันทีตั้งแต่เริ่มพูดหรือเปิดเพลง แต่ต้องรอครบ 15 วินาทีแรกก่อน

### Inference และ probability

function ที่ใช้ predict 1 window:

```python
process_window(...)
```

ข้างในทำ:

```text
waveform window
-> preprocessor
-> predictor.predict(...)
-> probabilities
```

โปรแกรมวัดเวลาด้วย:

```python
time.perf_counter()
```

แล้วแสดง:

```text
Inference: xx ms
RTF vs hop: x.xxx
```

RTF คือ real-time factor เทียบกับ hop:

```text
RTF = inference_time / hop_time
```

ถ้า `RTF < 1` แปลว่าเร็วพอสำหรับ realtime

### Soft vote 3 windows

ทุก window จะได้ probability ของ 8 genres เช่น:

```text
Electronic      0.08
Experimental    0.04
Folk            0.01
Hip-Hop         0.76
...
```

โปรแกรมเก็บ probability ล่าสุดไว้ใน:

```python
vote_buffer = deque(maxlen=vote_windows)
```

ค่า default:

```text
vote_windows = 3
```

จากนั้นเฉลี่ย probability:

```python
soft_probs = np.mean(vote_buffer, axis=0)
```

ผลที่แสดง realtime คือ genre ที่มีค่าเฉลี่ย probability สูงสุดจาก 3 windows ล่าสุด

ข้อดี:

- ผลนิ่งขึ้น
- ลดอาการ prediction แกว่งจาก window เดียว
- ยังตอบสนองเร็ว เพราะใช้แค่ 3 วินาทีล่าสุดของผล prediction

### Final majority vote

ตอนจบโปรแกรมเรียก:

```python
print_final_summary(...)
```

ผลสุดท้ายดูจาก:

```text
majority vote over soft decisions
```

แปลว่าแต่ละรอบ realtime ที่แสดงผลออกมา โปรแกรมจำไว้ว่า genre ที่ชนะคืออะไร แล้วตอนจบดูว่า genre ไหนชนะบ่อยที่สุด

ตัวอย่าง:

```text
Window 1  -> Hip-Hop
Window 2  -> Hip-Hop
Window 3  -> Pop
Window 4  -> Hip-Hop
```

final majority vote คือ:

```text
Hip-Hop
```

ใน summary โปรแกรมยังแสดงค่าอื่นให้เทียบด้วย:

- `Majority over soft votes` คือผลหลักที่ใช้เป็น final genre
- `Majority over raw windows` คือ majority จาก window เดี่ยว ๆ ไม่ผ่าน soft vote
- `All-window soft vote` คือเฉลี่ย probability ทุก window ทั้งเพลง
- `Top probabilities` คือ genre ที่มี probability เฉลี่ยสูงสุด

### Ensemble หลายโมเดล

ถ้าส่ง model หลายตัว:

```powershell
python realtime.py --model "models\a.pt" --model "models\b.pt" --file "songs\song.mp3"
```

โปรแกรมจะใช้:

```python
EnsemblePredictor
```

หลักการ:

1. ส่ง input เดียวกันเข้า model ทุกตัว
2. ได้ probability จากทุก model
3. เฉลี่ย probability ของทุก model
4. เอาค่าเฉลี่ยไปใช้กับ soft vote และ final vote ต่อ

โมเดลทุกตัวต้อง output จำนวน class เท่ากัน และ label order ต้องตรงกัน

### จุดที่แก้บ่อยถ้าจะพัฒนาต่อ

ถ้าจะเปลี่ยน folder model หรือเพลง:

```python
MODEL_EXTENSIONS
AUDIO_EXTENSIONS
apply_interactive_selection(...)
```

ถ้าจะเปลี่ยน preprocessing:

```python
AudioPreprocessor
```

ถ้าจะเพิ่ม architecture ใหม่:

```python
SpecAugCnn
build_builtin_model(...)
load_torch_backend(...)
```

ถ้าจะเปลี่ยนวิธี voting:

```python
vote_buffer
majority_vote(...)
print_final_summary(...)
```

ถ้าจะเปลี่ยนหน้าตา terminal display:

```python
display_results(...)
format_bar(...)
```

ถ้าจะเพิ่ม GUI:

```text
แนะนำให้ reuse logic เดิม:
- load_backends
- AudioPreprocessor
- run_from_file หรือ process_window
- print_final_summary อาจเปลี่ยนเป็นส่ง data กลับไปแสดงใน GUI
```

## PyTorch .pt กับ ONNX

ตอนนี้ยังไม่จำเป็นต้องแปลง `.pt` เป็น `.onnx` เพราะโมเดล `cnn_specaug_best.pt` รันเร็วพอสำหรับ realtime แล้ว

จากการทดสอบบน CPU:

- ใช้เวลาประมาณ `35-40 ms` ต่อ 1 prediction
- hop time คือ `1 second`
- ดังนั้น real-time factor ประมาณ `0.04`

เงื่อนไขสำคัญคือ:

```text
inference_time < hop_time
```

ตอนนี้:

```text
0.04s < 1.0s
```

จึงเพียงพอสำหรับ realtime demo

อย่างไรก็ตาม ถ้าต้องการ deploy จริง หรืออยาก optimize เพิ่ม สามารถ export เป็น ONNX และ benchmark เทียบกับ PyTorch ได้ในอนาคต

## คำสั่งที่มีประโยชน์

ดู options ทั้งหมด:

```powershell
python realtime.py --help
```

เลือกไฟล์เพลงผ่านเมนู:

```powershell
python realtime.py --mode file
```

ใช้ microphone:

```powershell
python realtime.py --mode mic
```

ระบุ model และเพลงเอง:

```powershell
python realtime.py --model "models\cnn_specaug_best.pt" --file "songs\my_song.mp3" --simulate-realtime
```

เปลี่ยนจำนวน windows ที่ใช้ soft vote:

```powershell
python realtime.py --model "models\cnn_specaug_best.pt" --file "songs\my_song.mp3" --vote-windows 5
```

ใช้หลายโมเดลแบบ ensemble:

```powershell
python realtime.py --model "models\model_a.pt" --model "models\model_b.pt" --file "songs\my_song.mp3"
```

ดู microphone devices:

```powershell
python realtime.py --list-devices
```

## Benchmark เพื่อเลือก model สำหรับ realtime

ถ้าต้องการเปรียบเทียบหลายโมเดลใน `models/` กับเพลงเดียวใน `songs/` ให้ใช้ไฟล์:

```text
benchmark_realtime_models.py
```

รันแบบเมนู:

```powershell
python benchmark_realtime_models.py
```

โปรแกรมจะให้เลือกหลายโมเดลได้ เช่น:

```text
all
1,3,4
2-5
```

จากนั้นเลือกเพลง 1 เพลงจาก `songs/` แล้วโปรแกรมจะประมวลผลทีละโมเดล และสรุปผลเป็นตาราง เช่น:

```text
Model
Final genre
Windows processed
Avg inference time
P95 inference time
Max inference time
RTL
P95 RTL
Ready
Top probability
```

นิยาม RTL ใน benchmark:

```text
RTL = average_inference_time / hop_time
```

ถ้า:

```text
RTL < 1
```

แปลว่าโมเดลประมวลผลทัน realtime ตาม hop ที่ตั้งไว้

ตัวอย่าง benchmark ทุกโมเดล:

```powershell
python benchmark_realtime_models.py --models all --file "000002.mp3"
```

ถ้าอยากเทสเร็ว ๆ แค่ไม่กี่ windows:

```powershell
python benchmark_realtime_models.py --models all --file "000002.mp3" --max-windows 3
```

ถ้าอยากเลือกเฉพาะบางโมเดล:

```powershell
python benchmark_realtime_models.py --models "cnn_specaug_best.pt,pond_best.pt,resnet_specaug_best.pt" --file "000002.mp3"
```

ถ้าอยาก export ผลเป็น CSV:

```powershell
python benchmark_realtime_models.py --models all --file "000002.mp3" --csv benchmark_results.csv
```

หมายเหตุ: `deep_cnn2d.onnx` เป็น ONNX เก่าที่ input shape ไม่ตรงกับ window 15 วินาที จึงอาจ fail ใน benchmark ปกติ ส่วน `ast_fma_best.pt` ใช้ได้แต่บน CPU จะช้ากว่าโมเดลอื่นมาก

## Dependencies

ควรใช้งานใน Python environment ที่มี package เหล่านี้:

```text
torch
torchaudio
numpy
soundfile
sounddevice
onnxruntime
transformers
```

ถ้าจะใช้ model ตระกูล EfficientNet ที่เซฟเป็น full PyTorch model เช่น `efficientnet_b0_full_model.pt` ต้องติดตั้งเพิ่ม:

```powershell
pip install timm
```

เหตุผลคือ full model `.pt` จะจำ Python class/library ตอน train ไว้ด้วย ถ้าตอนรันไม่มี library เดิม เช่น `timm` โปรแกรมจะโหลดโมเดลไม่ได้

ติดตั้งแบบพื้นฐาน:

```powershell
pip install -r requirements.txt
```

ถ้าใช้ conda environment เดิมที่ train/run โปรเจคนี้อยู่แล้ว อาจไม่ต้องติดตั้งเพิ่ม

## แนวทางพัฒนาต่อ

ไอเดียที่สามารถต่อยอดได้:

- ทำ GUI จริงด้วย Tkinter, PyQt, หรือ Streamlit
- เพิ่มปุ่ม start/stop recording
- แสดงกราฟ probability แบบ realtime
- บันทึกผล prediction เป็น `.csv`
- export model เป็น ONNX แล้ว benchmark ความเร็วกับ `.pt`
- เพิ่ม model ensemble แบบเลือกหลายโมเดลจากเมนู
- เพิ่ม confidence threshold ถ้าความมั่นใจต่ำให้แสดง `Unknown`
- เพิ่ม audio device selection สำหรับกรณีมีหลาย microphone
- ทำหน้า dashboard สำหรับสรุปผลทั้งเพลง เช่น genre timeline, top-k probability, majority vote count

## หมายเหตุสำหรับคนที่จะเอาไปใช้ต่อ

- ถ้าใช้ model ใหม่ที่ train กับ duration อื่น ต้องให้ `--window-sec` ตรงกับตอน train
- ถ้าใช้ checkpoint แบบ `state_dict` ที่ architecture ไม่เหมือน `cnn_specaug_best.pt` ต้องเพิ่ม model class หรือส่ง `--model-factory`
- ถ้า model output จำนวน class ไม่ตรงกับ genre list โปรแกรมจะแจ้ง error ให้ตรวจ label order
- ONNX model เก่า `deep_cnn2d.onnx` ในโปรเจคเดิมถูก export สำหรับ input shape คนละแบบ จึงไม่เหมาะกับ window 15 วินาที
