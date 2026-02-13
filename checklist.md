## Assignment Checklist

### Setup & Environment
- [X] Run `bash setup.sh` successfully
- [X] Verify model weights downloaded (stories42M.pt)

### Core Implementation
- [X] **llama.py**
- [X] **rope.py**
- [X] **optimizer.py**
- [X] **addition_data_generation.py**
- [X] **addition_run.py**

### Testing & Validation
- [X] Pass `python sanity_check.py` ## Run this for causal mask 
- [X] Pass `python optimizer_test.py` 
- [X] Pass `python rope_test.py` 
- [X] Generate coherent text with `python run_llama.py --option generate`
- [X] Complete SST zero-shot prompting
- [X] Complete CFIMDB zero-shot prompting  
- [X] Complete data generation for addition llama.
- [X] Complete five abalations. 
- [X] Achieve target accuracy 100% using a very tiny llama. 
- [X] Train your best model 

### Advanced Features (Optional - A+)
- [X] Create the tiniest adding llama. 

### Submission Preparation
- [X] Generate all required output files
- [X] Validate submission with `python prepare_submit.py /path/to/project/ ANDREWID`
- [X] Verify file size < 15MB
- [X] Create zip file with proper ANDREWID structure
- [X] Include optional report and feedback files
- [X] Final submission check before Canvas upload
