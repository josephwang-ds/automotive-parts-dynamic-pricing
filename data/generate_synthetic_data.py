"""命令行脚本：生成并保存合成数据。"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, N_SKUS, N_WEEKS, RANDOM_SEED
from src.data_generator import generate_all_data


def main():
    print(f"生成合成数据: {N_SKUS} SKUs, {N_WEEKS} 周...")
    products, sales = generate_all_data(N_SKUS, N_WEEKS, RANDOM_SEED)

    products_path = DATA_DIR / "synthetic_products.csv"
    sales_path = DATA_DIR / "synthetic_sales.csv"

    products.to_csv(products_path, index=False)
    sales.to_csv(sales_path, index=False)

    print(f"产品主数据: {len(products)} SKUs -> {products_path}")
    print(f"周度销售: {len(sales)} 行 -> {sales_path}")
    print(f"弹性范围: [{products['true_price_elasticity'].min():.3f}, {products['true_price_elasticity'].max():.3f}]")
    print(f"所有弹性为负: {(products['true_price_elasticity'] < 0).all()}")
    print("完成。")


if __name__ == "__main__":
    main()
