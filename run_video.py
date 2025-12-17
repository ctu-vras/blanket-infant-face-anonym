#!/usr/bin/env python3
import argparse
import sys
import warnings
from pathlib import Path

# suppress  warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spiga")
warnings.filterwarnings("ignore", category=UserWarning, module="controlnet_aux")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message=".*config attributes.*were passed.*but are not expected.*")

sys.path.insert(0, str(Path(__file__).parent))

from blanket.anonymization.pipelines.video_pipeline import VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="BLANKET Video Anonymization Pipeline (Hybrid Method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('video_path', help='Path to input video')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug frames with BB, CH overlay'
    )

    parser.add_argument(
        '--identity',
        help='Path to custom identity image'
    )

    parser.add_argument(
        '--identity-timestamp',
        type=float,
        help='Timestamp in seconds for identity frame '
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'mps', 'cpu'],
        help='Device for ML models (default: auto-detect)'
    )

    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual frames for manual inspection'
    )

    args = parser.parse_args()

    # Create output directory based on video name
    video_name = Path(args.video_path).stem
    output_dir = str(Path("output") / video_name)

    debug_dir = None
    if args.debug:
        debug_dir = str(Path(output_dir) / 'debug')

    # Print configuration
    print("=" * 60)
    print("BLANKET Video Anonymization Pipeline")
    print("=" * 60)
    print(f"Input:  {args.video_path}")
    print(f"Output: {output_dir}")
    if args.identity:
        print(f"Custom identity: {args.identity}")
    elif args.identity_timestamp is not None:
        print(f"Identity timestamp: {args.identity_timestamp}s")
    if debug_dir:
        print(f"Debug:  {debug_dir}")
    print("=" * 60)
    print()

    try:
        pipeline = VideoPipeline(
            output_dir=output_dir,
            debug_dir=debug_dir,
            face_detector_type="yolo",
            landmarks_detector_type="spiga",
            device=args.device,
            identity_image_path=args.identity,
            identity_timestamp=args.identity_timestamp,
            save_frames=args.save_frames,
            debug=args.debug,
        )

        result = pipeline.run(
            video_path=args.video_path,
        )
        return 0 if result['success'] else 1

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())