import torch
from modules.multimodal import Nivaya3Model
from modules.qch_subroutines import QuantumSubroutines

def test_nivaya_quantum():
    print("=== Testing Nivaya 3.0 (Quantum-Enhanced) ===")
    
    # 1. Initialize Model
    vocab_size = 1000
    embed_dim = 128
    model = Nivaya3Model(vocab_size, embed_dim, 2, 4, 2, 5)
    
    # 2. Forward Pass with Quantum Output
    batch_size = 1
    input_ids = torch.randint(0, vocab_size, (batch_size, 10))
    output = model(input_ids=input_ids)
    
    q_out = output['quantum_output']
    print(f"Quantum Delegation Triggered: {q_out['should_delegate'].item()}")
    print(f"Complexity Score: {q_out['complexity_score'].item():.4f}")
    
    # 3. Test VQE Subroutine
    print("\n[Test] Running VQE Molecular Simulation Subroutine...")
    angles = q_out['quantum_params']['angles'][0] # Take first batch
    energy = QuantumSubroutines.run_vqe_simulation(None, angles)
    print(f"Simulated Ground State Energy: {energy:.6f}")
    
    # 4. Test Error Mitigation
    print("\n[Test] Classical Error Mitigation (Post-processing)...")
    latent = output['last_hidden_state'][:, -1, :]
    mock_q_results = torch.randn(batch_size, 8) # Mock 8-qubit results
    cleaned = model.quantum_head.mitigate_errors(latent, mock_q_results)
    print(f"Cleaned Result Latent Shape: {cleaned.shape}")

    print("\n=== All Quantum-Enhanced Nivaya 3.0 Tests Passed! ===")

if __name__ == "__main__":
    test_nivaya_quantum()
