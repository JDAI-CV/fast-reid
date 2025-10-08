import os
import sys
import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, '.')

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer


class FastReIDConfigGenerator:
    
    DATASET_CONFIGS = {
        "Market1501": {
            "num_classes": 751,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        },
        "VeRiWild": {
            "num_classes": 30671,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        },
        "DukeMTMC": {
            "num_classes": 702,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        },
        "MSMT17": {
            "num_classes": 1041,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        },
        "CUHK03": {
            "num_classes": 767,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        },
        "VehicleID": {
            "num_classes": 13164,
            "input_size": [256, 128],
            "pixel_mean": [0.485, 0.456, 0.406],
            "pixel_std": [0.229, 0.224, 0.225]
        }
    }
    
    def detect_model_architecture(self, model_path: str) -> Dict[str, Any]:
        print(f"Analyzing model: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        layer_names = list(state_dict.keys())

        meta_arch = self._detect_meta_architecture(layer_names, model_path)
        backbone_info = self._detect_backbone(layer_names, state_dict)
        feature_dim = self._detect_feature_dim(state_dict)
        num_classes = self._detect_num_classes(state_dict)
        pooling_type = self._detect_pooling_type(layer_names)
        has_bnneck = self._detect_bnneck(layer_names)
        pixel_norm = self._extract_pixel_normalization(state_dict)
        
        arch_info = {
            "meta_architecture": meta_arch,
            "backbone_name": backbone_info["name"],
            "backbone_depth": backbone_info["depth"],
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "has_ibn": backbone_info["has_ibn"],
            "has_se": backbone_info["has_se"],
            "has_nl": backbone_info["has_nl"],
            "pooling_type": pooling_type,
            "has_bnneck": has_bnneck,
            "pixel_normalization": pixel_norm
        }
        
        print(f"Detected architecture: {arch_info}")
        return arch_info
    
    def _detect_meta_architecture(self, layer_names, model_path=None):
        # First check layer names for architecture-specific patterns
        if any('mgn' in name.lower() for name in layer_names):
            return 'MGN'

        # For now, map all other architectures to Baseline since they're not implemented
        # SBS, AGW, BOT are typically training techniques or variations of Baseline
        return 'Baseline'
    
    def _detect_backbone(self, layer_names, state_dict):
        has_ibn = any('.ibn.' in name or '.IN.' in name or 'ibn_' in name for name in layer_names)
        has_se = any('.se.' in name or 'squeeze' in name.lower() for name in layer_names)
        has_nl = any('NL_' in name or 'non_local' in name.lower() for name in layer_names)
        
        depth = self._detect_depth(layer_names)
        
        return {
            "name": "build_resnet_backbone",
            "depth": depth,
            "has_ibn": has_ibn,
            "has_se": has_se,
            "has_nl": has_nl
        }
    
    def _detect_depth(self, layer_names):
        # Count blocks in each layer to determine ResNet depth
        layer_counts = {}
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = set()
            for name in layer_names:
                if f'backbone.{layer_name}.' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == layer_name and i + 1 < len(parts):
                            try:
                                block_idx = int(parts[i + 1])
                                blocks.add(block_idx)
                            except ValueError:
                                pass
            layer_counts[layer_name] = len(blocks)

        # ResNet architectures have specific block counts:
        # ResNet-18: [2, 2, 2, 2]
        # ResNet-34: [3, 4, 6, 3]
        # ResNet-50: [3, 4, 6, 3]
        # ResNet-101: [3, 4, 23, 3]
        # ResNet-152: [3, 8, 36, 3]

        layer3_blocks = layer_counts.get('layer3', 0)
        layer2_blocks = layer_counts.get('layer2', 0)
        layer4_blocks = layer_counts.get('layer4', 0)

        print(f"   Layer block counts: {layer_counts}")

        if layer3_blocks >= 36:
            return '152x'
        elif layer3_blocks >= 23:
            return '101x'
        elif layer3_blocks >= 6:
            return '50x'
        elif layer3_blocks >= 2 and layer2_blocks >= 4:
            return '34x'
        elif layer3_blocks >= 2:
            return '18x'
        else:
            return '50x'  # Default fallback
    
    def _detect_feature_dim(self, state_dict):
        for name, tensor in state_dict.items():
            if ('heads' in name or 'classifier' in name) and 'weight' in name:
                if 'bnneck' in name and len(tensor.shape) >= 1:
                    return tensor.shape[0] if len(tensor.shape) == 1 else tensor.shape[1]
                elif 'classifier' in name and len(tensor.shape) == 2:
                    return tensor.shape[1]
        return 2048
    
    def _detect_num_classes(self, state_dict):
        for name, tensor in state_dict.items():
            if ('classifier' in name or 'cls_layer' in name) and 'weight' in name and len(tensor.shape) == 2:
                return tensor.shape[0]
        return None
    
    def _detect_pooling_type(self, layer_names):
        if any('gem' in name.lower() for name in layer_names):
            return 'gempoolP'
        elif any('attention' in name.lower() for name in layer_names):
            return 'AttentionPool'
        else:
            return 'GlobalAvgPool'
    
    def _detect_bnneck(self, layer_names):
        return any('bnneck' in name.lower() or 'bottleneck' in name.lower() for name in layer_names)
    
    def _extract_pixel_normalization(self, state_dict):
        pixel_norm = {}
        if 'pixel_mean' in state_dict:
            pixel_norm['mean'] = state_dict['pixel_mean'].squeeze().tolist()
        if 'pixel_std' in state_dict:
            pixel_norm['std'] = state_dict['pixel_std'].squeeze().tolist()
        return pixel_norm
    
    def _guess_dataset(self, num_classes):
        if num_classes is None:
            return "VeRiWild"
        
        for dataset, info in self.DATASET_CONFIGS.items():
            if info['num_classes'] == num_classes:
                return dataset
        return "VeRiWild"
    
    def generate_config(self, model_path: str, output_path: str, dataset: Optional[str] = None) -> str:
        arch_info = self.detect_model_architecture(model_path)
        
        if dataset is None:
            dataset = self._guess_dataset(arch_info['num_classes'])
        
        dataset_info = self.DATASET_CONFIGS.get(dataset, self.DATASET_CONFIGS["VeRiWild"])
        
        config = {
            "_META_": {
                "generated_by": "FastReID Config Generator v2",
                "model_name": Path(model_path).stem,
                "detected_architecture": arch_info['meta_architecture'],
                "detected_backbone": f"{arch_info['backbone_depth']}" + ("-IBN" if arch_info['has_ibn'] else "")
            },
            "MODEL": {
                "META_ARCHITECTURE": arch_info['meta_architecture'],
                "BACKBONE": {
                    "NAME": arch_info['backbone_name'],
                    "NORM": "BN",
                    "DEPTH": arch_info['backbone_depth'],
                    "LAST_STRIDE": 1,
                    "FEAT_DIM": arch_info['feature_dim'],
                    "WITH_IBN": arch_info['has_ibn'],
                    "WITH_SE": arch_info['has_se'],
                    "WITH_NL": arch_info['has_nl'],
                    "PRETRAIN": False
                },
                "HEADS": {
                    "NAME": "EmbeddingHead",
                    "NORM": "BN",
                    "WITH_BNNECK": arch_info['has_bnneck'],
                    "POOL_LAYER": arch_info['pooling_type'],
                    "NECK_FEAT": "before" if arch_info['has_bnneck'] else "after",
                    "CLS_LAYER": "Linear"
                },
                "LOSSES": {
                    "NAME": ["CrossEntropyLoss", "TripletLoss"],
                    "CE": {
                        "EPSILON": 0.1,
                        "SCALE": 1.0
                    },
                    "TRI": {
                        "MARGIN": 0.3,
                        "HARD_MINING": True,
                        "NORM_FEAT": False,
                        "SCALE": 1.0
                    }
                }
            },
            "INPUT": {
                "SIZE_TRAIN": dataset_info["input_size"],
                "SIZE_TEST": dataset_info["input_size"]
            },
            "DATASETS": {
                "NAMES": [dataset],
                "TESTS": [dataset]
            },
            "TEST": {
                "EVAL_PERIOD": 50,
                "IMS_PER_BATCH": 128,
                "METRIC": "cosine"
            },
            "CUDNN_BENCHMARK": True
        }
        
        if arch_info['pixel_normalization']:
            if 'mean' in arch_info['pixel_normalization']:
                config['MODEL']['PIXEL_MEAN'] = arch_info['pixel_normalization']['mean']
            if 'std' in arch_info['pixel_normalization']:
                config['MODEL']['PIXEL_STD'] = arch_info['pixel_normalization']['std']
        else:
            config['MODEL']['PIXEL_MEAN'] = [123.675, 116.28, 103.53]
            config['MODEL']['PIXEL_STD'] = [58.395, 57.12, 57.375]
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to: {output_path}")

        # Also create a clean version without metadata for direct use
        clean_config = {k: v for k, v in config.items() if k != '_META_'}
        clean_output_path = output_path.replace('.yml', '_clean.yml')
        with open(clean_output_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)

        print(f"Clean configuration (without metadata) saved to: {clean_output_path}")
        return output_path
    
    def validate_config(self, config_path: str, model_path: str) -> bool:
        import os
        import yaml

        try:
            print("Validating generated configuration...")

            # Load and clean the config for validation
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Remove metadata that's not part of FastReID config
            if '_META_' in config_data:
                del config_data['_META_']

            # Create a temporary config file for validation
            temp_config_path = config_path.replace('.yml', '_temp.yml')
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            cfg = get_cfg()
            cfg.merge_from_file(temp_config_path)
            cfg.MODEL.DEVICE = 'cpu'
            cfg.freeze()

            model = build_model(cfg)
            print("   Model building successful")

            if os.path.exists(model_path):
                checkpointer = Checkpointer(model)
                checkpointer.load(model_path)
                print("   Model weight loading successful")

            dummy_input = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
            model.eval()
            with torch.no_grad():
                inputs = {"images": dummy_input}
                output = model(inputs)
                print(f"   Inference test successful (output shape: {output.shape})")

            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            print("   Configuration validation passed")
            return True

        except Exception as e:
            print(f"   Configuration validation failed: {e}")
            # Clean up temp file on error
            temp_config_path = config_path.replace('.yml', '_temp.yml')
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return False
    
    def print_summary(self, config_path: str, model_path: str):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("\n" + "="*60)
            print("MODEL CONFIGURATION SUMMARY")
            print("="*60)
            print(f"Model File: {model_path}")
            print(f"Config File: {config_path}")
            print(f"Generated: {config.get('_META_', {}).get('generated_by', 'Unknown')}")
            
            model_config = config.get("MODEL", {})
            backbone_config = model_config.get("BACKBONE", {})
            heads_config = model_config.get("HEADS", {})
            input_config = config.get("INPUT", {})
            
            print(f"Meta Architecture: {model_config.get('META_ARCHITECTURE', 'Unknown')}")
            print(f"Backbone: {backbone_config.get('NAME', 'Unknown')}")
            print(f"Depth: {backbone_config.get('DEPTH', 'Unknown')}")
            print(f"Feature Dim: {backbone_config.get('FEAT_DIM', 'Unknown')}")
            print(f"IBN: {backbone_config.get('WITH_IBN', False)}")
            print(f"SE: {backbone_config.get('WITH_SE', False)}")
            print(f"Non-Local: {backbone_config.get('WITH_NL', False)}")
            print(f"Pooling: {heads_config.get('POOL_LAYER', 'Unknown')}")
            print(f"BNNeck: {heads_config.get('WITH_BNNECK', False)}")
            print(f"Input Size: {input_config.get('SIZE_TEST', 'Unknown')}")
            print(f"Dataset: {config.get('DATASETS', {}).get('NAMES', ['Unknown'])[0]}")
            
            detected = config.get('_META_', {}).get('detected_backbone', 'Unknown')
            print(f"Detected: {detected}")
            print("="*60)
            
        except Exception as e:
            print(f"Failed to print summary: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Improved FastReID Configuration Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_config.py --model market_sbs_R101-ibn.pth
  python generate_config.py --model veriwild_bot_R50-ibn.pth --dataset VeRiWild
  python generate_config.py --model model.pth --output custom_config.yml --no-validate
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pth)")
    parser.add_argument("--output", type=str, default="generated_config.yml", help="Output config file path")
    parser.add_argument("--dataset", type=str, help="Target dataset name (auto-detected if not specified)")
    parser.add_argument("--no-validate", action="store_true", help="Skip config validation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        print("Improved FastReID Configuration Generator")
        print("=" * 50)
        
        generator = FastReIDConfigGenerator()
        
        config_path = generator.generate_config(args.model, args.output, args.dataset)
        
        if not args.no_validate:
            success = generator.validate_config(config_path, args.model)
            if not success:
                print("Warning: Configuration validation failed, but file was generated.")
        
        generator.print_summary(config_path, args.model)
        
        print(f"\nConfiguration generated successfully!")
        print(f"Usage in your code:")
        print(f"  cfg = get_cfg()")
        print(f"  cfg.merge_from_file('{config_path}')")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()