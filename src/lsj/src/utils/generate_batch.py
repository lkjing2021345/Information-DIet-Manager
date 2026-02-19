#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import asyncio
from pathlib import Path
from smart_auto_generate import SmartAutoGenerator

async def main():
    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Use absolute path
    script_dir = Path(__file__).parent
    training_data_path = script_dir.parent / 'training_data' / 'training_data.json'
    
    generator = SmartAutoGenerator(
        api_key=config.get('api_key'),
        base_url=config.get('base_url'),
        model=config.get('model'),
        training_data_path=str(training_data_path)
    )
    
    # Generate in smaller batches to avoid timeout
    # Target: 10000 total, currently have ~5700, need ~4300 more
    # Generate 100 per category at a time
    
    for batch_num in range(1, 20):  # 7 batches of ~600 each
        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/20")
        print(f"{'='*60}")
        
        # Analyze current data
        current_counts = generator.analyze_existing_data()
        total_current = sum(current_counts.values())
        
        if total_current >= 10000:
            print(f"Target reached! Total: {total_current}")
            break
        
        # Generate 100 per category
        plan = {cat: 8 for cat in generator.CATEGORY_INFO.keys()}
        
        await generator.execute_generation_plan(plan)
        
        print(f"\nBatch {batch_num} completed. Waiting 5 seconds...")
        await asyncio.sleep(5)
    
    # Final analysis
    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    generator.analyze_existing_data()

if __name__ == "__main__":
    asyncio.run(main())
