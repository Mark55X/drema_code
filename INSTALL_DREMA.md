# Guida Completa all'Installazione di DreMa e VG-Mapping

Guida passo-passo per ricreare da zero l'ambiente funzionante per **`drema_code`** e **`vgmapping_drema`** utilizzando **`virtualenv`** con **Python 3.8** (testato e funzionante senza dover installare compilatori di sistema aggiuntivi).

---

## 1. Creazione e Attivazione del Virtualenv (Python 3.8)

Posizionati nella cartella principale del progetto:

```bash
cd ~/Master_Thesis

# Creazione dell'ambiente con Python 3.8
virtualenv -p python3.8 drema_env
# (oppure: python3.8 -m venv drema_env)

# Attivazione dell'ambiente
source drema_env/bin/activate
```

Aggiorna subito i tool essenziali e installa `ninja` (per velocizzare la compilazione CUDA):

```bash
pip install --upgrade pip setuptools wheel
pip install ninja
```

---

## 2. Installazione di PyTorch con Supporto CUDA

Installa la versione di PyTorch compatibile con i moduli di Gaussian Splatting:

```bash
# PyTorch 2.1.1 con CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

Verifica rapida della disponibilità GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 3. Preparazione e Compilazione Submoduli Gaussian Splatting

Entra nella directory `drema_code`:

```bash
cd ~/Master_Thesis/drema_code
```

### 3.1. Risoluzione Header `glm` Mancante
Il sotto-modulo `diff-surfel-rasterization` necessita degli header `glm` che sono già presenti negli altri submoduli. Copiali con:

```bash
cp -r submodules/diff-gaussian-rasterization/third_party/glm submodules/diff-surfel-rasterization/third_party/
```

### 3.2. Pulizia di Eventuali Build Precedenti
```bash
rm -rf submodules/*/build submodules/*/dist submodules/*/*.egg-info
```

### 3.3. Compilazione e Installazione dei 4 Submoduli
Compila i pacchetti C++/CUDA impostando i flag per `nvcc`:

```bash
# Permette a nvcc di compilare senza errori di compatibilità
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"

# Installazione dei 4 submoduli
pip install --no-cache-dir submodules/simple-knn
pip install --no-cache-dir submodules/diff-gaussian-rasterization
pip install --no-cache-dir submodules/diff-gaussian-rasterization-depth
pip install --no-cache-dir submodules/diff-surfel-rasterization
```

---

## 4. Installazione Dipendenze (`requirements.txt`)

> **Nota:** Nel `requirements.txt` originale c'era un conflitto tra `trimesh==3.10` e `viser>=0.2.0` (che richiede `trimesh>=3.21.7`). La versione corretta di `trimesh` è `trimesh>=3.21.7,<4.0.0`.

Installa tutte le dipendenze richieste:

```bash
cd ~/Master_Thesis/drema_code
pip install -r requirements.txt
```

*(In alternativa con comando diretto)*:
```bash
pip install opencv-python "numpy==1.23.5" open3d matplotlib object2urdf "trimesh>=3.21.7,<4.0.0" pybullet mediapy plyfile "hydra-core>=1.3.2" pynput "viser>=0.2.0" imageio imageio-ffmpeg
```

---

## 5. Installazione del Modulo `vgmapping_drema`

Installa il pacchetto `vgmapping_drema` in modalità *editable* (`-e`):

```bash
cd ~/Master_Thesis/vgmapping_drema
pip install -e .
```

---

## 6. Verifica del Funzionamento

Ritorna in `drema_code` ed esegui gli script di test per verificare che tutto sia operativo:

```bash
cd ~/Master_Thesis/drema_code

# 1. Test integrazione VG-Mapping Closed-Loop & RecurGS SE(3)
python test_drema_vg_mapping.py

# 2. Test simulazione
python simulate.py
```

---

## Riepilogo Rapido Tutti i Comandi (One-Liner / Script)

```bash
# 1. Ambiente
cd ~/Master_Thesis
virtualenv -p python3.8 drema_env
source drema_env/bin/activate
pip install --upgrade pip setuptools wheel ninja

# 2. PyTorch CUDA
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Submoduli GS
cd drema_code
cp -r submodules/diff-gaussian-rasterization/third_party/glm submodules/diff-surfel-rasterization/third_party/
rm -rf submodules/*/build submodules/*/dist submodules/*/*.egg-info
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
pip install --no-cache-dir submodules/simple-knn submodules/diff-gaussian-rasterization submodules/diff-gaussian-rasterization-depth submodules/diff-surfel-rasterization

# 4. Dipendenze DreMa
pip install -r requirements.txt

# 5. VG-Mapping
cd ../vgmapping_drema
pip install -e .
cd ../drema_code
```
