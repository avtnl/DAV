from loguru import logger
from file_manager import FileManager
from data_editor import DataEditor
from data_preparation import DataPreparation, InteractionSettings, NoMessageContentSettings
from plot_manager import PlotManager
import scripts

def main():
    SCRIPTS = [1,2,3]  # Only Script1 for testing

    file_manager = FileManager()
    data_editor = DataEditor()
    data_preparation = DataPreparation(
        data_editor=data_editor,
        int_settings=InteractionSettings(),
        nmc_settings=NoMessageContentSettings(),
    )
    plot_manager = PlotManager()

    # Default config matching config.toml
    default_config = {
        "raw": "data/raw",
        "processed": "data/processed",
        "image": "img",
        "input": "_chat.txt",
        "preprocess": False,
        "current_dac": "whatsapp-20250910-005041-dac-cleaned.parq",
        "current": "whatsapp-20250910-002822-maap-cleaned.parq",
        "raw_1": "_chat_maap.txt",
        "raw_2a": "_chat_vooranger_golfmaten.txt",
        "raw_2b": "_chat_golfmaten.txt",
        "raw_3": "_chat_dac.txt",
        "raw_4": "_chat_tillies.txt",
        "current_1": "whatsapp-20250910-002822-maap-cleaned.parq",
        "current_2a": "whatsapp-20250910-004727-golf-cleaned.parq",
        "current_2b": "whatsapp-20250910-012654-voorganger-golf-cleaned.parq",
        "current_3": "whatsapp-20250910-005041-dac-cleaned.parq",
        "current_4": "whatsapp-20250910-012135-til-cleaned.parq",
        "inputpath": "whatsapp-20250910-002822-maap-cleaned.csv",
        "datetime_format": "%d-%m-%Y, %H:%M:%S",
        "drop_authors": []
    }

    preprocess_script = scripts.Script0(
        file_manager, data_editor, data_preparation, None, default_config, None, scripts=SCRIPTS
    )
    logger.debug("Running Script0")
    result = preprocess_script.run()
    
    if result is None:
        logger.error("Preprocessing failed. Exiting.")
        return
    
    df = result['df']
    group_authors = result['group_authors']
    image_dir = result['image_dir']
    processed_dir = result['processed_dir']

    logger.debug(f"Preprocessed df shape: {df.shape}")
    logger.debug(f"Preprocessed df columns: {df.columns.tolist()}")
    logger.debug(f"Preprocessed df head:\n{df.head().to_string()}")
    logger.debug(f"group_authors: {group_authors}")
    logger.debug(f"image_dir: {image_dir}, processed_dir: {processed_dir}")

    script_list = {
        0: preprocess_script,
        1: scripts.Script1(file_manager, data_preparation, plot_manager, image_dir, df),
        2: scripts.Script2(file_manager, data_preparation, plot_manager, image_dir, df),
        3: scripts.Script3(file_manager, data_editor, data_preparation, plot_manager, image_dir, df),
        4: scripts.Script4(file_manager, data_preparation, plot_manager, image_dir, group_authors, df),
        5: scripts.Script5(file_manager, data_preparation, plot_manager, image_dir, df),
        7: scripts.Script7(file_manager, data_preparation, plot_manager, image_dir, group_authors, df),
        10: scripts.Script10(file_manager, data_editor, data_preparation, processed_dir, image_dir),
        11: scripts.Script11(file_manager, data_editor, data_preparation, plot_manager, processed_dir, image_dir),
    }

    for script_id in SCRIPTS:
        if script_id in script_list:
            logger.debug(f"Running script {script_id}")
            script_list[script_id].run()
        else:
            logger.warning(f"Unknown script {script_id} in Script list. Skipping.")

if __name__ == "__main__":
    main()