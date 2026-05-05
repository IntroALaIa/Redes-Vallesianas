from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define constants for state representation for better readability
STATE_NO = 0
STATE_YES = 1

def create_model() -> BayesianNetwork:
    """
    Crea y valida una Red Bayesiana para el ejemplo del aspersor.
    
    Returns:
        BayesianNetwork: El modelo configurado y validado.
    """
    model = BayesianNetwork([
        ('Nublado', 'Aspersor'),
        ('Nublado', 'Lluvia'),
        ('Aspersor', 'Suelo_Mojado'),
        ('Lluvia', 'Suelo_Mojado')
    ])

    # 1. Probabilidad de que esté nublado: [P(No), P(Sí)]
    cpd_nublado = TabularCPD(variable='Nublado', variable_card=2, values=[[0.5], [0.5]])

    # 2. Probabilidad del Aspersor dado Nublado: P(Aspersor | Nublado)
    # Filas: Aspersor (STATE_NO, STATE_YES)
    # Columnas: Nublado (STATE_NO, STATE_YES)
    cpd_aspersor = TabularCPD(variable='Aspersor', variable_card=2,
                                    # Nublado=STATE_NO, Nublado=STATE_YES
                              values=[[0.5, 0.9],  # Aspersor=STATE_NO
                                      [0.5, 0.1]], # Aspersor=STATE_YES
                              evidence=['Nublado'], evidence_card=[2])

    # 3. Probabilidad de Lluvia dado Nublado: P(Lluvia | Nublado)
    # Filas: Lluvia (STATE_NO, STATE_YES)
    # Columnas: Nublado (STATE_NO, STATE_YES)
    cpd_lluvia = TabularCPD(variable='Lluvia', variable_card=2, 
                            # Nublado=STATE_NO, Nublado=STATE_YES
                            values=[[0.8, 0.2],  # Lluvia=STATE_NO
                                    [0.2, 0.8]], # Lluvia=STATE_YES
                            evidence=['Nublado'], evidence_card=[2])

    # 4. Suelo Mojado dado Aspersor y Lluvia (Esta es la tabla más grande)
    # Filas: Suelo_Mojado (STATE_NO, STATE_YES)
    # Columnas (combinaciones de evidencia):
    # Aspersor=STATE_NO, Lluvia=STATE_NO
    # Aspersor=STATE_NO, Lluvia=STATE_YES
    # Aspersor=STATE_YES, Lluvia=STATE_NO
    # Aspersor=STATE_YES, Lluvia=STATE_YES
    cpd_suelo = TabularCPD(variable='Suelo_Mojado', variable_card=2,
                           #Asp=No, Lluv=NoAsp=No, Lluv=SíAsp=Sí, Lluv=NoAsp=Sí, Lluv=Sí
                           values=[[1.0, 0.1, 0.1, 0.01], # No mojado
                                   [0.0, 0.9, 0.9, 0.99]], # Mojado
                           evidence=['Aspersor', 'Lluvia'], evidence_card=[2, 2])

    # Agregamos las tablas al modelo
    model.add_cpds(cpd_nublado, cpd_aspersor, cpd_lluvia, cpd_suelo)
    
    # Validamos que todo sume 1 y las conexiones sean correctas (Grafo DAG, Grafo Acíclico Dirigido)
    if model.check_model():
        return model
    raise ValueError("El modelo no es válido. Verifique las CPDs.")

if __name__ == "__main__":
    # Crear el modelo
    modelo_aspersor = create_model()
    print("¡Modelo creado y validado con éxito!")

    # Ejemplo de inferencia: ¿Cuál es la probabilidad de que esté nublado si el suelo está mojado?
    inferencia = VariableElimination(modelo_aspersor)
    resultado = inferencia.query(
        variables=['Nublado'],
        evidence={'Suelo_Mojado': STATE_YES}  # Using constant for clarity
    )
    print("\nInferencia: P(Nublado | Suelo_Mojado = Sí)")
    print(resultado)