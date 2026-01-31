import pandas as pd
import numpy as np

def generate_sample_csv(filename="dados_credito.csv", n=1000):
    np.random.seed(42) # Seed
    
    # Média 600, Desvio 150, limitado entre 300 e 1000
    score = np.random.normal(600, 150, n).clip(300, 1000).astype(int)
    
    # Gerar Renda (Lognormal - mais realista para Brasil)
    renda = np.random.lognormal(8.5, 0.6, n).round(2)
    
    # Gerar Idade
    idade = np.random.randint(18, 75, n)
    
    # Definir Probabilidade Real de Default (A "Mágica")
    # Quanto maior o score, DRASTICAMENTE menor a chance de calote.
    # Fórmula: PD baseada no inverso do score ao quadrado (exponencial)
    # Adicionamos um pouco de aleatoriedade (ruído)
    prob_real = (1000 - score) ** 2 / 500000 
    prob_real = np.clip(prob_real, 0.01, 0.99) # Trava entre 1% e 99%
    
    # 5. Definir quem é inadimplente com base nessa probabilidade
    inadimplente = np.random.binomial(1, prob_real)
    
    # 6. Valor do Empréstimo
    # Quem ganha mais, tende a pedir mais, mas não linearmente
    valor_emprestimo = (renda * np.random.uniform(2, 12, n)).round(2)

    data = {
        'renda_mensal': renda,
        'score_serasa': score,
        'idade': idade,
        'valor_emprestimo': valor_emprestimo,
        'inadimplente': inadimplente
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    

    gini_estimado = abs(2 * (0.5 - np.mean(inadimplente)) * 100) # Proxy tosco só para debug
    print(f"✅ Arquivo gerado com {n} registros.")
    print(f"Taxa de Inadimplência da Carteira: {inadimplente.mean():.2%}")

if __name__ == "__main__":
    generate_sample_csv()
