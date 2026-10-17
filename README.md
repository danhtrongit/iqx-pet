# IQX Pixel Pets

Bản vẽ lại dạng **pixel art 512×512** của 4 linh thú IQX: Thanh Long, Lộc Hươu, Phụng Hoàng, Kim Quy.
Bám sát tranh gốc trong `danhtrongit/IQX` (`frontend/public/assets/mascots-2d/v2/<slug>/poster.webp`),
pixel cứng (không khử răng cưa, không pixel bán trong suốt), palette giới hạn, viền rõ.

| Thanh Long | Lộc Hươu | Phụng Hoàng | Kim Quy |
|---|---|---|---|
| ![](sprites/thanh-long.png) | ![](sprites/loc-huou.png) | ![](sprites/phung-hoang.png) | ![](sprites/kim-quy.png) |

## Thư mục

| Đường dẫn | Nội dung |
|---|---|
| `sprites/<slug>.png` | Sprite chính 512×512, PNG indexed (màu 0 trong suốt) |
| `dist/<slug>@2x.png` | Phóng 1024×1024 nearest-neighbor cho màn hình retina |
| `dist/compare/<slug>.png` | So sánh gốc / pixel |
| `dist/sheet.png` | 4 linh thú trên một hàng |
| `reference/<slug>.webp` | Tranh gốc 1024px dùng làm mẫu |
| `pets/<slug>.py` | Thông số + chỉnh tay riêng cho từng con |
| `tools/` | Pipeline chuyển đổi và công cụ soi |
| `preview.html` | Trang xem nhanh, bật/tắt bản gốc |

## Build

Cần Python 3 với `numpy` và `Pillow`.

```bash
python3 tools/build.py              # cả 4 con
python3 tools/build.py kim-quy      # một con
python3 tools/build.py --size 128   # bản pixel thô hơn (bỏ qua chỉnh tay), xem thử
```

Cách hoạt động (`tools/pixelize.py`):

1. Cắt sát nhân vật, thu nhỏ kiểu box về 512px.
2. Alpha cứng (≥50% là đặc) và một vòng viền 1px bên ngoài.
3. Các nét mực tối trong tranh gốc được giữ thành một màu viền duy nhất.
4. Giảm màu bằng k-means trong không gian Lab.
5. Lọc đa số 3×3 và gộp các mảng màu nhỏ để vùng màu phẳng, sạch.
6. `pets/<slug>.py` → `touchup(art)` vẽ lại bằng tay những chi tiết pipeline làm hỏng
   (chữ IQX, đá quý, mắt, má hồng, râu, gạc...).

## Chỉnh sửa

Soi một vùng (toạ độ sprite 0–511) cạnh bản gốc, có lưới toạ độ:

```bash
python3 tools/inspect_region.py loc-huou 180 120 330 240 --zoom 4 --out /tmp/mat.png
```

Rồi sửa `pets/<slug>.py`: `CONFIG` để đổi thông số pipeline, `touchup(art)` để vẽ tay
(`art.put`, `art.rect`, `art.ellipse`, `art.line`, `art.replace`, `art.blit`, hoặc sửa thẳng `art.idx`).
