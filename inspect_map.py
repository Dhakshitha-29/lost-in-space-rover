#!/usr/bin/env python3
"""inspect_map.py - Inspection tool for generated Ground Truth planetary maps.

Usage:
    python inspect_map.py
    python inspect_map.py --seed 42
    python inspect_map.py --seed 123 --save-image map.png
    python inspect_map.py --width 40 --height 40 --seed 99
"""

import argparse
import os
import sys

from rover_sim.config import SimulationConfig, TerrainType
from rover_sim.world import WorldGenerator


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect procedurally generated planetary worlds for Lost in Space."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic world generation (default: 42)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=40,
        help="Grid width in cells (default: 40)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=40,
        help="Grid height in cells (default: 40)",
    )
    parser.add_argument(
        "--comm-zones",
        type=int,
        default=3,
        help="Number of communication zones (default: 3)",
    )
    parser.add_argument(
        "--data-sites",
        type=int,
        default=8,
        help="Number of data sites (default: 8)",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Optional path to save map image (e.g. map.png)",
    )
    parser.add_argument(
        "--no-ascii",
        action="store_true",
        help="Skip printing the ASCII grid and print only summary statistics",
    )
    parser.add_argument(
        "--rover",
        action="store_true",
        help="Display the Rover's initial Belief Map and sensing status at Base",
    )
    return parser.parse_args()


def render_image(world, output_path: str) -> None:
    """Save graphical visualization of the world using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless backend
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"[!] Matplotlib is required to export images: pip install matplotlib")
        return

    w, h = world.width, world.height

    # Color code terrain:
    # 0 = Flat (light green / sage)
    # 1 = Rough (sand / tan)
    # 2 = Hill (brown / terracotta)
    # 3 = Rock (dark charcoal / black)
    color_matrix = np.zeros((h, w, 3), dtype=float)
    color_palette = {
        TerrainType.FLAT: (0.78, 0.88, 0.72),   # Flat plains (light sage)
        TerrainType.ROUGH: (0.92, 0.78, 0.55),  # Rough sand (warm tan)
        TerrainType.HILL: (0.72, 0.50, 0.38),   # Hill terrain (rust brown)
        TerrainType.ROCK: (0.22, 0.22, 0.25),   # Impassable rock (dark charcoal)
    }

    for y in range(h):
        for x in range(w):
            cell = world.get_cell(x, y)
            color_matrix[y, x] = color_palette[cell.terrain]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(color_matrix, origin="upper")

    # Plot Base
    bx, by = world.base_pos
    ax.plot(bx, by, marker="s", markersize=14, color="#1E88E5", markeredgecolor="white", markeredgewidth=1.5)
    ax.text(bx, by, "B", color="white", fontsize=8, fontweight="bold", ha="center", va="center")

    # Plot Comm Zones
    for cx, cy in world.comm_zones:
        ax.plot(cx, cy, marker="D", markersize=12, color="#00ACC1", markeredgecolor="white", markeredgewidth=1.5)
        ax.text(cx, cy, "C", color="white", fontsize=7, fontweight="bold", ha="center", va="center")

    # Plot Data Sites
    for ds_pos, ds in world.data_sites.items():
        is_highest = ds.id == world.highest_value_site.id
        if is_highest:
            ax.plot(ds.x, ds.y, marker="*", markersize=18, color="#FFD700", markeredgecolor="#B71C1C", markeredgewidth=1.5)
            ax.text(ds.x, ds.y - 1.0, f"{ds.name} ({ds.value})", color="#B71C1C", fontsize=8, fontweight="bold", ha="center")
        else:
            ax.plot(ds.x, ds.y, marker="o", markersize=11, color="#FB8C00", markeredgecolor="white", markeredgewidth=1.5)
            ax.text(ds.x, ds.y, str(ds.id), color="white", fontsize=7, fontweight="bold", ha="center", va="center")

    ax.set_title(
        f"Ground Truth Planetary Map (Seed {world.seed}, {w}x{h})\n"
        f"Base: {world.base_pos} | Comm Zones: {len(world.comm_zones)} | Data Sites: {len(world.data_sites)}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks(range(0, w, 5))
    ax.set_yticks(range(0, h, 5))
    ax.grid(True, color="gray", alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend patches
    legend_elements = [
        mpatches.Patch(color=color_palette[TerrainType.FLAT], label="Flat (Cost: 1)"),
        mpatches.Patch(color=color_palette[TerrainType.ROUGH], label="Rough (Cost: 3)"),
        mpatches.Patch(color=color_palette[TerrainType.HILL], label="Hill (Cost: 5)"),
        mpatches.Patch(color=color_palette[TerrainType.ROCK], label="Rock / Crater (Cost: inf)"),
        plt.Line2D([0], [0], marker="s", color="w", label="Rover Base (B)", markerfacecolor="#1E88E5", markersize=10),
        plt.Line2D([0], [0], marker="D", color="w", label="Comm Zone (C)", markerfacecolor="#00ACC1", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Data Site (D)", markerfacecolor="#FB8C00", markersize=10),
        plt.Line2D([0], [0], marker="*", color="w", label=f"Highest Site ({world.highest_value_site.value} pts)", markerfacecolor="#FFD700", markeredgecolor="#B71C1C", markersize=14),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Graphical map saved successfully to: {output_path}")


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()

    config = SimulationConfig(
        grid_width=args.width,
        grid_height=args.height,
        seed=args.seed,
        num_comm_zones=args.comm_zones,
        num_data_sites=args.data_sites,
    )

    print(f"Generating planetary world with seed {args.seed} ({args.width}x{args.height})...")
    world = WorldGenerator.generate(config)

    if not args.no_ascii:
        print("\n" + world.render_ascii())

    world.print_summary()

    if args.rover:
        from rover_sim.rover import Rover
        rover = Rover.create_at_base(world, config=config)
        print("\n" + "=" * 60)
        print("[+] ROVER INITIAL BELIEF MAP (After Landing at Base)")
        print("=" * 60)
        print(rover.render_ascii_belief())
        status = rover.status()
        print("\nRover Status:")
        print(f"  * Position: {status['position']}")
        print(f"  * Energy: {status['energy']} / {status['max_energy']} ({status['energy_pct']})")
        print(f"  * Sensor Radius: {status['sensor_radius']} (Manhattan distance)")
        print(f"  * Safety Margin: {rover.safety_margin * 100:.0f}% (Buffer: {rover.safety_margin * rover.max_energy:.1f} energy units)")
        print(f"  * Known Cells: {status['known_count']} / {config.grid_width * config.grid_height}")
        print(f"  * Frontier Cells: {status['frontier_count']}")
        print("  * Current Terrain Cost Estimates:")
        for t in [TerrainType.FLAT, TerrainType.ROUGH, TerrainType.HILL]:
            print(f"      - {t.value.capitalize()}: {rover.get_estimated_cost(t):.2f}")

    if args.save_image:
        render_image(world, args.save_image)


if __name__ == "__main__":
    main()
