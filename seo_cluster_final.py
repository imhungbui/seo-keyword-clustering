import os
import sys
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime


def parse_volume(vol_str):
    """Parse volume từ string, xử lý dấu chấm/phẩy ngăn cách hàng nghìn."""
    if not vol_str:
        return 0
    # Bỏ tất cả ký tự không phải số (dấu chấm, phẩy, khoảng trắng, ký tự lạ)
    vol_clean = re.sub(r'[^\d]', '', str(vol_str).strip())
    return int(vol_clean) if vol_clean else 0


def split_line_smart(line):
    """
    Tự động phát hiện và tách dòng theo nhiều định dạng:
    - Tab: "keyword\t123"        (copy từ Excel/Google Sheets)
    - Dấu phẩy: "keyword,123"    (CSV)
    - Dấu chấm phẩy: "keyword;123"
    - Nhiều khoảng trắng: "keyword    123"
    - Volume ở cuối dòng (số cuối cùng)
    """
    line = line.strip()
    if not line:
        return None, None

    # Ưu tiên 1: Có tab → tách theo tab (format chuẩn từ Excel/Sheets)
    if '\t' in line:
        parts = line.split('\t')
        keyword = parts[0].strip()
        volume = parse_volume(parts[-1]) if len(parts) > 1 else 0
        return keyword, volume

    # Ưu tiên 2: Có dấu chấm phẩy
    if ';' in line:
        parts = line.rsplit(';', 1)
        return parts[0].strip(), parse_volume(parts[1]) if len(parts) > 1 else 0

    # Ưu tiên 3: Tìm số ở cuối dòng (có thể đứng sau dấu phẩy hoặc khoảng trắng)
    match = re.search(r'^(.*?)[\s,]+([\d.,]+)\s*$', line)
    if match:
        keyword = match.group(1).strip().rstrip(',').strip()
        volume = parse_volume(match.group(2))
        return keyword, volume

    # Không tìm thấy volume → coi cả dòng là keyword, volume = 0
    return line, 0


def process_input(raw_text):
    """Parse input đa định dạng (tab, comma, semicolon, hoặc space)."""
    parsed_data = []
    lines = raw_text.strip().split('\n')
    for line in lines:
        keyword, volume = split_line_smart(line)
        if keyword:
            parsed_data.append({'keyword': keyword, 'volume': volume})
    return pd.DataFrame(parsed_data)


def run_clustering(df, similarity_threshold=0.88):
    """Gom cụm keyword bằng embedding model đa ngôn ngữ."""
    print("\n[1/3] Đang tải AI Multilingual E5 (hỗ trợ cả tiếng Anh & tiếng Việt)...")
    model = SentenceTransformer('intfloat/multilingual-e5-base')

    print(f"[2/3] Đang phân tích ngữ nghĩa {len(df)} từ khóa...")
    queries_for_ai = ["query: " + kw for kw in df['keyword'].tolist()]

    embeddings = model.encode(
        queries_for_ai,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    distance_threshold = 1 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='complete',
        distance_threshold=distance_threshold
    )
    df['cluster_id'] = clustering.fit_predict(embeddings)
    return df


def export_to_excel(df_clustered):
    """Xuất kết quả ra file Excel trong thư mục Downloads."""
    print("\n[3/3] Đang đóng gói dữ liệu và xuất file Excel...")
    final_data = []

    for cluster_id, group in df_clustered.groupby('cluster_id'):
        group = group.sort_values(by='volume', ascending=False)
        keywords_list = group.to_dict('records')
        primary = keywords_list[0]

        for kw in keywords_list:
            final_data.append({
                'Chủ đề chính (Main Topic)': primary['keyword'],
                'Tổng Volume của Cụm': int(group['volume'].sum()),
                'Số từ khóa trong cụm': len(keywords_list),
                'Từ khóa (Keyword)': kw['keyword'],
                'Volume': kw['volume']
            })

    df_export = pd.DataFrame(final_data)
    df_export = df_export.sort_values(
        by=['Tổng Volume của Cụm', 'Volume'],
        ascending=[False, False]
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Keyword_Clusters_{now}.xlsx"

    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.isdir(downloads_dir):
        downloads_dir = os.path.expanduser("~")
    output_path = os.path.join(downloads_dir, filename)

    df_export.to_excel(output_path, index=False)
    return output_path, len(df_export['Chủ đề chính (Main Topic)'].unique())


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🌏 TOOL GOM CỤM TỪ KHÓA AI - ĐA NGÔN NGỮ (EN + VI) 🌏")
    print("=" * 55)
    print("📝 Bước 1: Copy keyword + volume (từ Excel, Sheets, Ahrefs, SEMrush...).")
    print("           Hỗ trợ mọi định dạng: tab, dấu phẩy, hoặc khoảng trắng.")
    print("📝 Bước 2: Dán (Cmd + V / Ctrl + V) vào cửa sổ terminal này.")
    print("⚠️  Bước 3: Sau khi dán xong, nhấn [Enter] 1 lần,")
    print("           rồi nhấn [Ctrl + D] (Mac/Linux) hoặc [Ctrl + Z] + [Enter] (Windows).")
    print("-" * 55 + "\n")

    try:
        raw_input_text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\n❌ Đã hủy thao tác.")
        sys.exit()

    if not raw_input_text.strip():
        print("❌ Bạn chưa nhập dữ liệu. Hãy chạy lại lệnh nhé!")
        sys.exit()

    df = process_input(raw_input_text)

    if df.empty:
        print("❌ Không tìm thấy keyword hợp lệ trong dữ liệu đầu vào.")
        sys.exit()

    print(f"\n✓ Đã nhận {len(df)} từ khóa hợp lệ.")
    print(f"  Ví dụ: \"{df.iloc[0]['keyword']}\" (volume: {df.iloc[0]['volume']})")

    df_clustered = run_clustering(df, similarity_threshold=0.88)

    output_path, num_clusters = export_to_excel(df_clustered)

    print("\n" + "=" * 55)
    print(f"✅ THÀNH CÔNG!")
    print(f"📊 Đã gom {len(df)} từ khóa thành {num_clusters} cụm chủ đề.")
    print(f"📂 File: {output_path}")
    print("=" * 55 + "\n")
