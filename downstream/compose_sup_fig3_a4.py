from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


PROJECT_ROOT = Path("/data/user/hesy/projects/SpatialMETA")
FIGURE_DIR = PROJECT_ROOT / "SpaDTA_718/runs/sm_downstream/sup_fig3/figures"
OUTPUT_PATH = FIGURE_DIR / "sup_fig3_A4_width_3quarter_length.png"

DPI = 600
PAGE_WIDTH = round(210 / 25.4 * DPI)
PAGE_HEIGHT = round((297 * 0.75) / 25.4 * DPI)
MARGIN = 24
GAP = 18

PANELS = {
    "a": FIGURE_DIR / "a_spadta_cluster12_cluster8_on_slice.png",
    "b": FIGURE_DIR / "b_cluster12_vs_cluster8_differential_genes.png",
    "c": FIGURE_DIR / "c_spatial_deg_expression.png",
    "d": FIGURE_DIR / "d_selected_genes_expression_boxplots.png",
    "e": FIGURE_DIR / "e_cluster12_vs_cluster8_differential_metabolites.png",
    "f": FIGURE_DIR / "f_spatial_metabolite_abundance.png",
    "g": FIGURE_DIR / "g_selected_metabolites_abundance_boxplots.png",
    "h": FIGURE_DIR / "h_cluster12_cluster8_multiomics_enrichment.png",
}

# column, row, displayed panel label
LAYOUT = {
    "a": (0, 0, "a"),
    "h": (1, 0, "b"),
    "b": (0, 1, "c"),
    "e": (1, 1, "d"),
    "d": (0, 2, "e"),
    "g": (1, 2, "f"),
    "c": (0, 3, "g"),
    "f": (1, 3, "h"),
}


def trim_white(image: Image.Image, padding: int = 8) -> Image.Image:
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    difference = difference.convert("L").point(lambda value: 255 if value > 8 else 0)
    box = difference.getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def fit_to_cell(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except OSError:
        font = ImageFont.load_default()

    cell_width = (PAGE_WIDTH - 2 * MARGIN - GAP) / 2
    cell_height = (PAGE_HEIGHT - 2 * MARGIN - 3 * GAP) / 4

    for source_label, source in PANELS.items():
        column, row, display_label = LAYOUT[source_label]
        x = round(MARGIN + column * (cell_width + GAP))
        y = round(MARGIN + row * (cell_height + GAP))
        width = round(cell_width)
        height = round(cell_height)

        image = fit_to_cell(trim_white(Image.open(source)), width, height)
        image_x = x
        image_y = y
        canvas.paste(image, (image_x, image_y))

        label_x = image_x + 5
        label_y = image_y + 5
        label_box = (label_x, label_y, label_x + 58, label_y + 62)
        draw.rounded_rectangle(label_box, radius=4, fill="white")
        draw.text((label_x + 7, label_y - 1), display_label, fill="black", font=font)

    canvas.save(OUTPUT_PATH, dpi=(DPI, DPI), compress_level=5)


if __name__ == "__main__":
    main()
