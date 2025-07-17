# LocalRAG

this application automates loading,processing PDF data in /data and response generation from a custom locally run LLM(mistral or phi-2)

Steps:

1.Make sure u hv python 3.10 or 3.11 for unerrotic execution.

2.Change Python interpreter to 3.11 or 3.10 and run "py -3.11 -m pip install -r requirements.txt"(or 3.10)

3.run case "CPU": "pip uninstall torch torchvision torchaudio -y"
      case "CUDA 11.8" : "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
      case "CUDA 12.1" : "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
      run "nvidia-smi" in shell or cmd to get the version

4.Install mistrall if u has GPU's VRAM is more than 8GB else phi2 from installation files in /installations

microsoft/phi-2 was used for testing, it takes 2-3 min to respond with a RTX 3050 4GB GPU

5.load data with pdf files(within a limit and make sure content is readable) and run "py -3.11 update_db.py"

4.Now to get LLM answers to questions, run "py -3.11 query_rag.py "-->question""

Note: The above takes a lot of time and can be errotic, so feel free to skip steps 1-3 and comment out code under "#online" in query_rag.py and Embeddings.py and use "ollama pull --model" on a different terminal
                                          "ollama run -->model"