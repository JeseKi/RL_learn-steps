"""
文件功能：
- 统一的 matplotlib 延迟导入与中文字体全局设置。

公开接口：
- _ensure_matplotlib(): 返回 (plt, font_prop, title_font_prop)

内部方法：
- 无。

公开接口的 pydantic 模型：
- 无。
"""

from pathlib import Path


def _ensure_matplotlib():
    """延迟导入 matplotlib 并全局设置中文字体，返回 (plt, font_prop, title_font_prop)。

    数据流：
    - 尝试加载 assets/微软雅黑.ttf；若存在则注册并设为 rcParams 全局字体；
    - 回退默认字体，并提示可能出现方框。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 非交互后端
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        from matplotlib import font_manager as fm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"未安装 matplotlib 或当前环境不可用：{e}. 请通过 conda 安装 matplotlib。"
        )

    font_path = Path.cwd() / "assets" / "微软雅黑.ttf"
    if font_path.exists():
        try:
            fm.fontManager.addfont(str(font_path))
        except Exception:
            pass
        fp_tmp = FontProperties(fname=str(font_path))
        font_name = fp_tmp.get_name() or "Microsoft YaHei"
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        font_prop = FontProperties(fname=str(font_path), size=12)
        title_font_prop = FontProperties(fname=str(font_path), size=16)
        print(f"✅ 字体文件 {font_path} 已加载")
    else:
        font_prop = FontProperties(size=12)
        title_font_prop = FontProperties(size=16)
        print(f"⚠️ 未找到中文字体文件 {font_path}，将使用默认字体（可能出现方框）。")
    return plt, font_prop, title_font_prop
