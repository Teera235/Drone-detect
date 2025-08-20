
import argparse
import sys
import codecs

from config import DetectionConfig, DisplayConfig, GuidanceConfig, SOFTWARE_INFO
from system import OrdnanceDroneDetectionSystem

def build_parser():
    p = argparse.ArgumentParser(
        description=f"{SOFTWARE_INFO['name']} v{SOFTWARE_INFO['version']} - Enhanced UI"
    )
    p.add_argument('--model','-m', default='best.pt')
    p.add_argument('--confidence','-c', type=float, default=0.5)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--classes', nargs='+', default=['drone','aircraft','helicopter','uav'])
    p.add_argument('--save-dir', default='detections')
    p.add_argument('--auto-save', action='store_true', default=True)
    p.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    p.add_argument('--fov-h', type=float, default=62.2)
    p.add_argument('--fov-v', type=float, default=48.8)
    p.add_argument('--roi-margin', type=float, default=0.08)
    p.add_argument('--roi-scale', type=float, default=1.0, help='Centered ROI size ratio (0<r<=1). 0.5 = half-size')
    p.add_argument('--guide-all', action='store_true')
    return p

def main():
    try:
        if sys.platform.startswith('win'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except: pass

    args = build_parser().parse_args()

    det = DetectionConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        target_classes=args.classes,
        auto_save_detections=args.auto_save,
        save_directory=args.save_dir
    )
    disp = DisplayConfig()
    guide = GuidanceConfig(
        fov_deg_h=args.fov_h, fov_deg_v=args.fov_v,
        roi_margin_ratio=args.roi_margin,
        roi_size_ratio=args.roi_scale,
        show_only_highest_conf=(not args.guide_all)
    )

    sysm = OrdnanceDroneDetectionSystem(det, disp, guide, log_level=args.log_level)
    sysm.run()

if __name__ == "__main__":
    main()
## python main.py -m best.pt --roi-scale 0.6
