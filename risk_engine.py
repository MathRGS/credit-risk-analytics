import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Configuração de Logs para auditoria de execução
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CreditRiskEngine:
    def __init__(self, model_path='credit_model.pkl', lgd=0.45):
        self.model_path = model_path
        self.lgd = lgd  # Loss Given Default (Basileia III Std)
        self.model = LogisticRegression(class_weight='balanced', solver='liblinear')
        self.features = ['renda_mensal', 'score_serasa', 'idade']

    def load_dataset(self, filepath):
        #Carregamento e validação de integridade dos dados.
        try:
            df = pd.read_csv(filepath)
            # Validação simples de schema
            missing = [col for col in self.features + ['inadimplente'] if col not in df.columns]
            if missing:
                raise ValueError(f"Colunas ausentes no CSV: {missing}")
            
            logger.info(f"Dataset carregado: {len(df)} registros para processamento.")
            return df
        except Exception as e:
            logger.error(f"Falha no carregamento: {str(e)}")
            raise

    def train_and_evaluate(self, df):
        # Pipeline de treino com cálculo de Gini e métricas de classificação.
        X = df[self.features]
        y = df['inadimplente']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        logger.info("Iniciando treinamento do modelo logístico...")
        self.model.fit(X_train, y_train)

        # Avaliação de Performance (Gini é padrão em Risco de Crédito)
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        gini = 2 * auc - 1
        
        logger.info(f"Modelo Treinado. AUC: {auc:.3f} | Gini Coefficient: {gini:.2%}")
        
        # Salvando artefato do modelo (MLOps)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo salvo em disco: {self.model_path}")

        return X_test, y_test

    def calculate_portfolio_risk(self, df, X_test):
        """
        Cálculo vetorial de Perda Esperada (EL).
        """
        # Recupera o EAD (Valor do Empréstimo) correspondente ao conjunto de teste
        portfolio = df.loc[X_test.index].copy()
        
        # Inferência de PD (Probability of Default)
        portfolio['pd'] = self.model.predict_proba(X_test)[:, 1]
        
        # Cálculo: EL = PD * LGD * EAD
        # Uso de np.multiply para garantir operação vetorizada e eficiente
        portfolio['expected_loss'] = np.multiply(
            np.multiply(portfolio['pd'], self.lgd), 
            portfolio['valor_emprestimo']
        )

        return portfolio

    def generate_executive_report(self, portfolio):
        print("\n" + "="*60)
        print(f"RELATÓRIO EXECUTIVO DE RISCO DE CRÉDITO")
        print("="*60)
        
        total_ead = portfolio['valor_emprestimo'].sum()
        total_el = portfolio['expected_loss'].sum()
        avg_pd = portfolio['pd'].mean()
        
        print(f"Volume Analisado (N):     {len(portfolio)} contratos")
        print(f"Exposição Total (EAD):    R$ {total_ead:,.2f}")
        print(f"Perda Esperada (EL):      R$ {total_el:,.2f}")
        print(f"Índice de Cobertura (EL%):{(total_el/total_ead):.2%}")
        print(f"Prob. Default Média (PD): {avg_pd:.2%}")
        print("-" * 60)
        print("Distribuição de Risco (Head 5):")
        print(portfolio[['score_serasa', 'valor_emprestimo', 'pd', 'expected_loss']].head())
        print("="*60)

if __name__ == "__main__":
    engine = CreditRiskEngine()
    
    # Execução do Pipeline
    try:
        raw_data = engine.load_dataset('dados_credito.csv')

        engine.train_and_evaluate(raw_data)

        X_full = raw_data[engine.features]
        full_portfolio = engine.calculate_portfolio_risk(raw_data, X_full)

        engine.generate_executive_report(full_portfolio)
        
    except Exception as main_error:
        logger.critical(f"Processo abortado: {main_error}")
