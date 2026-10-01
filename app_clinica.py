import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai

# 1. CONFIGURACIÓN DE INTELIGENCIA ARTIFICIAL
mi_clave = st.secrets["GEMINI_API_KEY"] 
genai.configure(api_key=mi_clave)
modelo_ia = genai.GenerativeModel('gemini-3.8-flash')

# 2. CONFIGURACIÓN DE RUTAS Y MODELO
ruta_base = './'
ruta_modelo = ruta_base + 'modelo_pronostico_ibero.pkl'
archivo_base_datos = ruta_base + 'registro_pacientes.csv'
modelo_ortopedia = joblib.load(ruta_modelo)

# 3. INTERFAZ VISUAL
st.title("🏥 Asistente Predictivo de Ortopedia")
st.write("Cálculo de Return to Play (RTP) con respaldo bibliográfico de IA.")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad del paciente", min_value=12, max_value=90, value=26)
    meses_evolucion = st.number_input("Meses de evolución", min_value=0, max_value=120, value=2)
    deportista = st.selectbox("¿Es deportista activo?", ["Sí", "No"])
    tratamiento = st.selectbox("Manejo propuesto", ["Conservador", "Quirúrgico"])

with col2:
    tabaquismo = st.selectbox("¿Tabaquismo?", ["Sí", "No"])
    articulacion = st.selectbox("Articulación afectada", ["Rodilla", "Hombro"])
    
    if articulacion == "Rodilla":
        opciones_dx = ["Lesion Meniscal", "Ruptura LCA", "LCA + Menisco", "Esquina Posterolateral"]
    else:
        opciones_dx = ["Lesion SLAP", "Inestabilidad Bankart", "Lesion Supraespinoso", "Lesion Infraespinoso", "Lesion Subescapular", "Manguito Rotador Masivo"]
        
    diagnostico = st.selectbox("Diagnóstico principal", opciones_dx)

# 4. EVALUACIÓN Y RESPALDO CIENTÍFICO
if st.button("Evaluar Paciente y Consultar Literatura"):
    
    # Preparación de datos para el modelo estadístico
    es_deportista = 1 if deportista == "Sí" else 0
    fuma = 1 if tabaquismo == "Sí" else 0
    es_rodilla = 1 if articulacion == "Rodilla" else 0
    es_hombro = 1 if articulacion == "Hombro" else 0
    
    columnas_completas = [
        'Edad', 'Deportista_Activo', 'Meses_Evolucion', 'Tabaquismo', 
        'Articulacion_Hombro', 'Articulacion_Rodilla', 
        'Diagnostico_Esquina Posterolateral', 'Diagnostico_Inestabilidad Bankart', 
        'Diagnostico_LCA + Menisco', 'Diagnostico_Lesion Infraespinoso', 
        'Diagnostico_Lesion Meniscal', 'Diagnostico_Lesion SLAP', 
        'Diagnostico_Lesion Subescapular', 'Diagnostico_Lesion Supraespinoso', 
        'Diagnostico_Manguito Rotador Masivo', 'Diagnostico_Ruptura LCA'
    ]
    
    paciente_nuevo = pd.DataFrame(columns=columnas_completas)
    paciente_nuevo.loc[0] = 0 
    
    paciente_nuevo['Edad'] = edad
    paciente_nuevo['Deportista_Activo'] = es_deportista
    paciente_nuevo['Meses_Evolucion'] = meses_evolucion
    paciente_nuevo['Tabaquismo'] = fuma
    paciente_nuevo['Articulacion_Rodilla'] = es_rodilla
    paciente_nuevo['Articulacion_Hombro'] = es_hombro
    paciente_nuevo[f'Diagnostico_{diagnostico}'] = 1

    # Predicción base matemática
    prediccion_base = modelo_ortopedia.predict(paciente_nuevo)[0]
    
    # Ajuste clínico preliminar (puedes refinar estos números luego)
    if diagnostico == "Lesion Meniscal" and tratamiento == "Quirúrgico":
        prediccion_final = prediccion_base - 3.0  
    elif diagnostico == "Ruptura LCA" and tratamiento == "Conservador":
        prediccion_final = prediccion_base + 12.0 
    elif tratamiento == "Quirúrgico":
        prediccion_final = prediccion_base + 4.0
    else:
        prediccion_final = prediccion_base - 1.0
        
    prediccion_final = max(1.0, prediccion_final)
    
    st.success(f"⏱️ Pronóstico ajustado: {prediccion_final:.1f} semanas para Return to Play.")
    
    # Consulta en vivo a Gemini para justificar el ajuste
    with st.spinner("Consultando literatura ortopédica para respaldar el tratamiento..."):
        instruccion = f"""
        Eres un cirujano articular especialista. 
        Un paciente presenta {diagnostico} y se ha elegido un tratamiento {tratamiento}.
        Redacta un párrafo de 3 a 4 líneas justificando con la literatura médica actual por qué este 
        tratamiento específico acelera o retrasa el Return to Play deportivo comparado con otras opciones. 
        Menciona el consenso general brevemente.
        """
        
        respuesta_ia = modelo_ia.generate_content(instruccion)
        
        # Mostramos la justificación bibliográfica debajo del resultado
        st.info(f"📚 **Respaldo Clínico (Literatura Actual):**\n\n{respuesta_ia.text}")

    # Guardado en base de datos
    paciente_nuevo['Tratamiento_Elegido'] = tratamiento
    paciente_nuevo['Semanas_Predichas'] = prediccion_final
    
    if not os.path.exists(archivo_base_datos):
        paciente_nuevo.to_csv(archivo_base_datos, index=False)
    else:
        paciente_nuevo.to_csv(archivo_base_datos, mode='a', header=False, index=False)
