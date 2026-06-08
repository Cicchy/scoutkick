from scoutkick.backend.src.services.pipeline_service import EPAPipeline

if __name__ == "__main__":
    try:
        pipeline = EPAPipeline("2024")
        engine = pipeline.run()
        if engine:
            sample_team = list(engine.epas.keys())[0]
            print(f"Sample Team {sample_team} Mean Total: {engine.get_team(sample_team).mean[0]:.2f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
