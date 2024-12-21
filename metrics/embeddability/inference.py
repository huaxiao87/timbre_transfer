import json
import torch

import os
import psutil

# Ottieni il PID del processo corrente
# pid = os.getpid()
# print("PID del processo python:", pid)

# # Impostare la priorità (maggiore è il valore negativo, maggiore è la priorità)
# psutil.Process(pid).nice(-12)  # Priorità alta

# # # Impostare l'affinità della CPU (eseguire solo sul core 0 e 1)
# # psutil.Process(pid).cpu_affinity([0, 1, 2])

# # Verifica l'affinità attuale
# print("CPU Affinity:", psutil.Process(pid).cpu_affinity())
# print("Process Nice:", psutil.Process(pid).nice())

# Costruisce un modello dinamico dai parametri JSON
def build_model(decoder_config, synth_config, ddsp_mode="hpn"):
    from DDSP.models.ddsp_decoder import DDSP_Decoder
    print(ddsp_mode)
    if ddsp_mode == "hpn":
        from DDSP.models.decoder.hpn_decoder import HpNdecoder
        from DDSP.models.synths.hpn_synth import HpNsynth
        decoder = HpNdecoder(
                            hidden_size=decoder_config["hidden_size"], 
                            num_layers=decoder_config["num_layers"],
                            input_keys=decoder_config["input_keys"],
                            input_sizes=decoder_config["input_sizes"],
                            output_keys=decoder_config["output_keys"],
                            output_sizes=decoder_config["output_sizes"]
                            )
        synth = HpNsynth(   
                         sample_rate=synth_config["sample_rate"],
                         block_size=synth_config["block_size"],
                         reverb_scale=synth_config["reverb_scale"]
                         )
        
    elif ddsp_mode == "wavetable":
        from DDSP.models.decoder.hpn_decoder import HpNdecoder
        from DDSP.models.synths.wavetable_synth import WTpNsynth
        decoder = HpNdecoder(
                            hidden_size=decoder_config["hidden_size"], 
                            num_layers=decoder_config["num_layers"],
                            input_keys=decoder_config["input_keys"],
                            input_sizes=decoder_config["input_sizes"],
                            output_keys=decoder_config["output_keys"],
                            output_sizes=decoder_config["output_sizes"]
                            )
        synth = WTpNsynth(   
                         sample_rate=synth_config["sample_rate"],
                         block_size=synth_config["block_size"],
                         reverb_scale=synth_config["reverb_scale"]
                         )
        
    elif ddsp_mode == "ddx7":
        from DDSP.models.decoder.ddx7_decoder import FMdecoder
        from DDSP.models.synths.ddx7_synth import FMsynth
        
        decoder = FMdecoder(
                    n_blocks=decoder_config["n_blocks"],
                    hidden_channels=decoder_config["hidden_channels"],
                    output_keys=['ol'],
                    out_channels=decoder_config["out_channels"],
                    kernel_size=decoder_config["kernel_size"],
                    dilation_base=decoder_config["dilation_base"],
                    apply_padding=decoder_config["apply_padding"],
                    deploy_residual=decoder_config["deploy_residual"],
                    input_keys=decoder_config["input_keys"],
                    z_size=None,
                    output_complete_controls=True
                    )
        
        synth = FMsynth(
                    sample_rate=synth_config["sample_rate"],
                    block_size=synth_config["block_size"],
                    max_ol=synth_config["max_ol"],
                    fr=synth_config["fr"],
                    synth_module=synth_config["synth_module"]
                    )
    else:
        raise ValueError(f"Configurazione non corretta.")

    # Return the DDSP_Decoder
    return DDSP_Decoder(decoder=decoder, synth=synth)


def main(args):
    config_path = args.config
    verbose = args.verbose
    
    # Carica il file di configurazione
    with open(config_path, 'r') as file:
        config = json.load(file)

    if "hpn" in config_path:
        ddsp = "hpn"
    elif "wavetable" in config_path:
        ddsp = "wavetable"
    elif "ddx7" in config_path:
        ddsp = "ddx7"
    else:
        raise ValueError(f"Invalid Configuration PATH. {config_path}")
    
    if "full" in config_path:
        version = "full"
    elif "reduced" in config_path:
        version = "reduced"
    else:
        raise ValueError(f"Invalid Configuration PATH. {config_path}")
    
    print(f"ANALISI RTF di modello: {ddsp.capitalize()} - {version.capitalize()} ")
    print("="*10)
    print(config_path)
    
    # Estrae i parametri del modello
    decoder_config = config['decoder']
    synth_config = config['synth']

    # Costruisce il modello dinamico
    model = build_model(decoder_config, synth_config, ddsp_mode=ddsp)
    if verbose:
        print("Model Created")
        print(model)

    model.eval()

    # Memory Usage (Load into RAM )
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    if param_memory > 100000:
        param_memory_mb = param_memory / (1024 ** 2)  # Convert to MB
        print(f"Model {ddsp.upper()} ({version.capitalize()}) Parameters Memory: {param_memory_mb:.2f} MB")
    else:
        param_memory_kb = param_memory / (1024)  # Convert to MB
        print(f"Model {ddsp.upper()} ({version.capitalize()}) Parameters Memory: {param_memory_kb:.2f} KB")

    
    
    import torch.utils.benchmark as benchmark
    
    # x= {}
    # buffer = 64000
    # x['audio'] = torch.randn(1, buffer, 1)
    # x['f0'] = torch.randn(1, int(buffer/64), 1)
    # x['f0_scaled'] = torch.randn(1, int(buffer/64), 1)
    # x['loudness_scaled'] = torch.randn(1, int(buffer/64), 1)
    # t0 = benchmark.Timer(
    #         stmt='model(x)',
    #         globals={'x': x, 'model': model},
    #         num_threads=1,
    #         label=ddsp.upper()+"_"+version.capitalize(),
    #         sub_label='4 seconds',
    #         description='teacher',
    #     )
    # print(t0.timeit(100))

    results = []
    x= {}
    
    def inference(net, y):
        with torch.no_grad():
            net(y)

    label = ddsp.upper()+"_"+version.capitalize()
    
    if args.multithread:
            threads = [1, 2, 4]
    else:
            threads = [1]
    
    # for buffer in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64000]:
    for buffer in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:

        x['audio'] = torch.randn(1, buffer, 1)
        x['f0'] = torch.randn(1, int(buffer/64), 1)
        x['f0_scaled'] = torch.randn(1, int(buffer/64), 1)
        x['loudness_scaled'] = torch.randn(1, int(buffer/64), 1)
        sub_label = f'buffer={buffer} - {buffer/16:.2f} ms'
        
        
        for num_threads in threads:
            print(f"Running with buffer={buffer} and num_thread={num_threads}...")
            results.append(benchmark.Timer(
                stmt='inference(model,x)',
                globals={'x': x, 'model': model, "inference": inference},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='timeit(100)',
                ).timeit(100)    
            )
            
            results.append(benchmark.Timer(
                stmt='inference(model,x)',
                globals={'x': x, 'model': model, "inference": inference},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='blocked_autorange',
                ).blocked_autorange()    
            )
    compare = benchmark.Compare(results)
    #compare.trim_significant_figures()
    #compare.colorize()
    compare.print()

if __name__ == "__main__":
    import argparse
    # Parser degli argomenti
    parser = argparse.ArgumentParser(description="Script per analizzare il RTF di un modello DDSP.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path al file di configurazione JSON",
    )
    
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose",
    )
    
    parser.add_argument(
        "--multithread",
        type=bool,
        default=False,
        help="Verbose",
    )
        
    args = parser.parse_args()

    # Script con il file di configurazione specificato
    main(args)
