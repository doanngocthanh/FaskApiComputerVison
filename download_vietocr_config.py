#!/usr/bin/env python3
"""
Script để download VietOCR config qua proxy và lưu vào file
"""

import os
import requests
import yaml
import json
from urllib.parse import urlparse

def download_vietocr_config():
    """Download VietOCR config qua proxy"""
    
    # Proxy configuration
    proxy_url = "http://cuongbo:bohem2805@160.30.112.35:3128"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    # VietOCR config URLs - try both base and vgg_transformer
    config_urls = [
        "https://vocr.vn/data/vietocr/config/base.yml",
        "https://vocr.vn/data/vietocr/config/vgg_transformer.yml"
    ]
    
    for i, config_url in enumerate(config_urls):
        config_name = "base" if i == 0 else "vgg_transformer"
        print(f"\n🌐 Downloading VietOCR config from: {config_url}")
        print(f"🔄 Using proxy: {proxy_url}")
        
        try:
            # Download config with proxy
            response = requests.get(config_url, proxies=proxies, timeout=30, verify=False)
            response.raise_for_status()
            
            print(f"✅ {config_name} config downloaded successfully")
            
            # Parse YAML
            config_data = yaml.safe_load(response.text)
            print(f"✅ {config_name} config parsed successfully")
            
            # Print config structure
            print(f"\n📋 {config_name} config structure:")
            for key, value in config_data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey in value.keys():
                        print(f"    - {subkey}")
                else:
                    print(f"  {key}: {type(value).__name__}")
            
            # Save to files
            yaml_file = f"vietocr_config_{config_name}.yml"
            json_file = f"vietocr_config_{config_name}.json"
            
            # Save as YAML
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            print(f"💾 Config saved to: {yaml_file}")
            
            # Save as JSON for easier Python integration
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Config saved to: {json_file}")
            
            # Print specific sections we need
            print(f"\n🔍 Key sections for {config_name} config:")
            
            if 'cnn' in config_data:
                print("CNN config:")
                print(json.dumps(config_data['cnn'], indent=2))
            else:
                print("❌ No CNN config found")
            
            if 'transformer' in config_data:
                print("\nTransformer config:")
                print(json.dumps(config_data['transformer'], indent=2))
                
            if 'backbone' in config_data:
                print(f"\nBackbone: {config_data['backbone']}")
            else:
                print("❌ No backbone found")
                
            if 'seq_modeling' in config_data:
                print(f"Seq modeling: {config_data['seq_modeling']}")
            
        except requests.exceptions.ProxyError as e:
            print(f"❌ Proxy error for {config_name}: {e}")
        except requests.exceptions.SSLError as e:
            print(f"❌ SSL error for {config_name}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error for {config_name}: {e}")
        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error for {config_name}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error for {config_name}: {e}")
    
    return True

def test_proxy_connection():
    """Test proxy connection"""
    proxy_url = "http://cuongbo:bohem2805@160.30.112.35:3128"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    test_url = "http://httpbin.org/ip"
    
    print(f"🧪 Testing proxy connection to: {test_url}")
    
    try:
        response = requests.get(test_url, proxies=proxies, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Proxy connection successful")
        print(f"📍 Your IP through proxy: {result.get('origin', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Proxy connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 VietOCR Config Downloader")
    print("=" * 50)
    
    # Test proxy first
    if test_proxy_connection():
        print("\n" + "=" * 50)
        config = download_vietocr_config()
        
        if config:
            print("\n✅ All done! Check the generated files:")
            print("  - vietocr_config.yml")
            print("  - vietocr_config.json")
        else:
            print("\n❌ Failed to download config")
    else:
        print("\n❌ Cannot proceed without working proxy connection")
