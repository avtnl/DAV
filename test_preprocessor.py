# test_preprocessor.py
import wa_analyzer.preprocess as preprocessor
from src.constants import PreprocessorArgs

print("Calling preprocessor.main()")
preprocessor.main([PreprocessorArgs.DEVICE, PreprocessorArgs.IOS])
print("This line will never print if sys.exit() is called")