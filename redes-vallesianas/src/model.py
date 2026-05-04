from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def create_model():
    model = BayesianNetwork([
        ('Nublado', 'Aspersor'),
        ('Nublado', 'Lluvia'),
        ('Aspersor', 'Suelo_Mojado'),
        ('Lluvia', 'Suelo_Mojado')
    ])

    # 1. Probabilidad de que esté nublado: [P(No), P(Sí)]
    cpd_nublado = TabularCPD(variable='Nublado', variable_card=2, values=[[0.5], [0.5]])

    # 2. Probabilidad del Aspersor dado Nublado: P(Aspersor | Nublado)
    # Filas: Aspersor (No, Sí) | Columnas: Nublado (No, Sí)
    cpd_aspersor = TabularCPD(variable='Aspersor', variable_card=2, 
                              values=[[0.5, 0.9],  # No
                                      [0.5, 0.1]], # Sí
                              evidence=['Nublado'], evidence_card=[2])

    # 3. Probabilidad de Lluvia dado Nublado: P(Lluvia | Nublado)
    cpd_lluvia = TabularCPD(variable='Lluvia', variable_card=2, 
                            values=[[0.8, 0.2],  # No
                                    [0.2, 0.8]], # Sí
                            evidence=['Nublado'], evidence_card=[2])

    # 4. Suelo Mojado dado Aspersor y Lluvia (Esta es la tabla más grande)
    cpd_suelo = TabularCPD(variable='Suelo_Mojado', variable_card=2,
                           values=[[1.0, 0.1, 0.1, 0.01], # No mojado
                                   [0.0, 0.9, 0.9, 0.99]], # Mojado
                           evidence=['Aspersor', 'Lluvia'], evidence_card=[2, 2])

    # Agregamos las tablas al modelo
    model.add_cpds(cpd_nublado, cpd_aspersor, cpd_lluvia, cpd_suelo)
    
    # Validamos que todo sume 1 y las conexiones sean correctas
    if model.check_model():
        return model

if __name__ == "__main__":
    red_lista = create_model()
    print("¡Modelo creado y validado con éxito!")