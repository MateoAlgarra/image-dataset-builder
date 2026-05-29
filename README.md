# 🖼️ AI Image Dataset Builder

> Automatically collects, filters, deduplicates and organizes image datasets ready for ML training (TensorFlow / PyTorch compatible).

**by [Mateo Algarra](https://github.com/MateoAlgarra)**

---

## ✨ Features

- 🔍 Scrapes images from any webpage automatically
- 📐 Filters by minimum resolution (default 224×224 — ML standard)
- 🧹 Removes duplicate images using MD5 hash detection
- 📁 Organizes images into labeled category folders
- 📄 Generates a CSV manifest (filename, class label, resolution, source URL, MD5)
- ⏱️ Smart random delays to avoid rate limiting / IP blocks
- 🔄 Saves progress — resumes where it left off if interrupted
- ✅ Output ready for TensorFlow, PyTorch, and Keras pipelines

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install pillow beautifulsoup4 requests tqdm colorama

# Run
python image_dataset_builder.py \
  --sources "https://example.com/cats" "https://example.com/dogs" \
  --categories "cats" "dogs" \
  --output my_dataset \
  --min-width 224 \
  --min-height 224 \
  --max-per-cat 500
```

---

## 📂 Output Structure

```
my_dataset/
├── cats/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── dogs/
│   ├── image_0001.jpg
│   └── ...
└── manifest_20260101_1200.csv
```

---

## 📋 CSV Manifest Format

| filename | class_label | width | height | source_url | md5 | collected_at |
|---|---|---|---|---|---|---|
| cats/image_0001.jpg | cats | 512 | 512 | https://... | a1b2c3... | 20260101_1200 |

---

## ⚙️ Arguments

| Argument | Default | Description |
|---|---|---|
| `--sources` | required | Page URLs to scrape (one per category) |
| `--categories` | required | Label for each URL |
| `--output` | `dataset_output` | Output folder |
| `--min-width` | `224` | Minimum image width in pixels |
| `--min-height` | `224` | Minimum image height in pixels |
| `--max-per-cat` | `100` | Max images per category |

---

## 💼 Use Cases

- Building training datasets for computer vision models
- Collecting product images for e-commerce AI
- Gathering labeled medical/satellite/retail imagery
- Any project requiring organized, clean image datasets

---

## 🛠️ Tech Stack

`Python` · `BeautifulSoup4` · `Pillow` · `requests` · `tqdm` · `colorama`

---

## 📬 Contact

Need a custom dataset pipeline?
**[hire me on Freelancer](https://www.freelancer.com/u/MATEOAIGARRA)**