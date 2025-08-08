from train.mlops_pipeline import MLOpsPipeline

def main():
    pipeline = MLOpsPipeline()
    
    try:
        if pipeline.initialize_services():
            pipeline.start_api_server()
        else:
            print("[ERROR] Service initialization failed.")
            
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
