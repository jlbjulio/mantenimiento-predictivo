# 🚀 Guía de Deploy en Streamlit Cloud

## Requisitos Previos

1. **Cuenta de GitHub** (gratuita)
2. **Cuenta de Streamlit Cloud** (gratuita) - https://share.streamlit.io/
3. **Git instalado** en tu computadora

---

## 📋 Paso 1: Preparar el Proyecto

### 1.1 Inicializar Git (si no lo has hecho)

```powershell
cd c:\Users\julio\Downloads\ProyectoFinalSI
git init
git add .
git commit -m "Initial commit - Sistema de Mantenimiento Predictivo"
```

### 1.2 Verificar Archivos Necesarios

✅ **requirements.txt** - Ya existe
✅ **.gitignore** - Creado (excluye logs temporales)
✅ **.streamlit/config.toml** - Creado (configuración de tema)
✅ **app/streamlit_app.py** - Aplicación principal
✅ **models/** - Modelos entrenados

---

## 📤 Paso 2: Subir a GitHub

### 2.1 Crear Repositorio en GitHub

1. Ve a https://github.com/new
2. Nombre: `mantenimiento-predictivo` (o el que prefieras)
3. Descripción: `Sistema Inteligente de Recomendación para Mantenimiento Predictivo`
4. Público o Privado (ambos funcionan)
5. **NO** inicialices con README (ya tienes uno)
6. Click en **"Create repository"**

### 2.2 Conectar y Subir

```powershell
# Agregar remote (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/mantenimiento-predictivo.git

# Renombrar branch a main (si es necesario)
git branch -M main

# Subir código
git push -u origin main
```

---

## 🌐 Paso 3: Deploy en Streamlit Cloud

### 3.1 Acceder a Streamlit Cloud

1. Ve a https://share.streamlit.io/
2. Click en **"Sign in"** o **"Sign up"**
3. Inicia sesión con tu cuenta de GitHub

### 3.2 Crear Nueva App

1. Click en **"New app"**
2. Configurar:
   - **Repository**: Selecciona `TU_USUARIO/mantenimiento-predictivo`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
3. Click en **"Deploy!"**

### 3.3 Esperar Deployment

- Proceso toma 2-5 minutos
- Streamlit Cloud instalará dependencias automáticamente
- Verás logs en tiempo real

---

## ⚙️ Paso 4: Configuración Post-Deploy

### 4.1 Verificar que Todo Funciona

1. La app debe cargar automáticamente
2. Verifica que los modelos se cargan correctamente
3. Haz una predicción de prueba
4. Verifica que el sistema de feedback funciona

### 4.2 URL de tu Aplicación

Tu app estará disponible en:

```
https://TU_USUARIO-mantenimiento-predictivo-app-streamlit-app-HASH.streamlit.app
```

Puedes personalizar la URL en la configuración de la app.

---

## 🔧 Problemas Comunes y Soluciones

### Error: "Module not found"

**Causa**: Falta una dependencia en `requirements.txt`

**Solución**:

```powershell
# Agregar dependencia faltante
echo "nombre-paquete==version" >> requirements.txt
git add requirements.txt
git commit -m "Agregar dependencia faltante"
git push
```

### Error: "File not found: models/"

**Causa**: Los modelos no se subieron a GitHub

**Solución**:

```powershell
# Verificar que .gitignore NO excluye models/
# Agregar y subir modelos
git add models/
git commit -m "Agregar modelos entrenados"
git push
```

### Error: "Memory limit exceeded"

**Causa**: Streamlit Cloud tiene límite de 1GB RAM (plan gratuito)

**Soluciones**:

1. Optimizar modelos (usar modelos más pequeños)
2. Reducir `n_estimators` en RandomForest
3. Considerar upgrade a plan de pago

### La App es Muy Lenta

**Causa**: Carga de modelos en cada ejecución

**Solución**: Ya implementado con `@st.cache_resource` en tu código

---

## 🔄 Actualizar la Aplicación

Cada vez que hagas cambios:

```powershell
# Hacer cambios en el código
# ...

# Guardar cambios
git add .
git commit -m "Descripción de los cambios"
git push

# Streamlit Cloud detecta cambios y redespliega automáticamente
```

---

## 📊 Gestión de Datos en Producción

### Logs y Feedback

⚠️ **Importante**: Los datos en Streamlit Cloud son **efímeros**

**Problema**:

- `logs/predicciones.csv` se reinicia cada vez que la app redespliega
- Feedback de usuarios se pierde

**Soluciones**:

#### Opción 1: Base de Datos Externa (Recomendado)

```python
# Usar PostgreSQL, MongoDB, o Firebase
# Ejemplo con Streamlit Secrets:
import streamlit as st
db_connection = st.secrets["connections"]["postgresql"]
```

#### Opción 2: Google Sheets API

```python
# Guardar feedback en Google Sheets
# Requiere configurar API key en Streamlit Secrets
```

#### Opción 3: AWS S3 / Google Cloud Storage

```python
# Guardar archivos CSV en bucket cloud
# Requiere credenciales en Streamlit Secrets
```

#### Opción 4: Solo Desarrollo/Demo

Si es solo para demostración:

- Los logs se reinician → no es problema
- Usuarios prueban funcionalidad sin persistencia

---

## 🔐 Secrets y Variables de Entorno

Si necesitas API keys o credenciales:

### En Streamlit Cloud:

1. Ve a tu app en https://share.streamlit.io/
2. Click en **"⚙️ Settings"**
3. Click en **"Secrets"**
4. Agregar en formato TOML:

```toml
[database]
host = "tu-host.com"
user = "tu-usuario"
password = "tu-password"

[api_keys]
openai = "sk-..."
```

### En tu código:

```python
import streamlit as st

# Acceder a secrets
db_host = st.secrets["database"]["host"]
api_key = st.secrets["api_keys"]["openai"]
```

---

## 📈 Monitoreo y Analytics

### Ver Logs de la App

1. En Streamlit Cloud, click en tu app
2. Click en **"Manage app"**
3. Ver **"Logs"** en tiempo real

### Métricas de Uso

- Streamlit Cloud muestra visitas y uso
- Considera Google Analytics para tracking avanzado

---

## 💰 Planes de Streamlit Cloud

### Plan Gratuito (Community)

- ✅ 1 app pública
- ✅ 1 GB RAM
- ✅ 1 CPU compartido
- ✅ Perfecto para demos y portafolio

### Plan de Pago (Team/Enterprise)

- 🚀 Apps ilimitadas
- 🚀 Más recursos (RAM/CPU)
- 🚀 Apps privadas
- 🚀 Soporte prioritario

---

## ✅ Checklist Pre-Deploy

- [ ] Código funciona localmente
- [ ] `requirements.txt` completo
- [ ] `.gitignore` configurado
- [ ] Modelos entrenados incluidos
- [ ] README.md con instrucciones
- [ ] Código subido a GitHub
- [ ] Cuenta de Streamlit Cloud creada
- [ ] App desplegada y funcionando

---

## 🎯 Siguientes Pasos

1. **Personalizar URL**: En settings de Streamlit Cloud
2. **Agregar logo**: Usar `st.logo()` en el código
3. **Custom domain**: Disponible en planes de pago
4. **Implementar persistencia**: Usar base de datos externa
5. **Agregar analytics**: Google Analytics o Mixpanel
6. **Compartir**: Enviar URL a usuarios/stakeholders

---

## 📞 Soporte

- **Documentación Streamlit**: https://docs.streamlit.io/
- **Foro de Streamlit**: https://discuss.streamlit.io/
- **GitHub Issues**: Para problemas del código

---

## 🌟 Tips Adicionales

### Mejorar Performance

```python
# Ya implementado en tu código:
@st.cache_resource
def load_model():
    return joblib.load('models/failure_binary_model.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('ai4i2020.csv')
```

### Agregar Página de Inicio Profesional

```python
# Agregar al inicio de streamlit_app.py
st.set_page_config(
    page_title="Mantenimiento Predictivo AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Habilitar Analytics

```python
# Agregar Google Analytics en HTML personalizado
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
""", unsafe_allow_html=True)
```

---

**¡Listo! Tu aplicación de Mantenimiento Predictivo está en la nube! 🎉**
