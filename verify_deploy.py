"""
Script de verificación pre-deploy para Streamlit Cloud
Verifica que todos los requisitos estén listos para deployment
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, required=True):
    """Verifica si un archivo existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filepath}")
    return exists

def check_directory_exists(dirpath, required=True):
    """Verifica si un directorio existe"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {dirpath}/")
    return exists

def check_git_initialized():
    """Verifica si Git está inicializado"""
    exists = os.path.exists('.git')
    status = "✅" if exists else "❌"
    print(f"{status} Repositorio Git inicializado")
    return exists

def check_requirements():
    """Verifica que requirements.txt tenga los paquetes necesarios"""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt no encontrado")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read().lower()
    
    required_packages = [
        'streamlit', 'pandas', 'scikit-learn', 'joblib', 'shap',
        'numpy', 'plotly', 'altair', 'fastapi', 'uvicorn',
        'python-dateutil', 'pydantic'
    ]
    missing = []
    
    for pkg in required_packages:
        if pkg not in content:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Faltan paquetes en requirements.txt: {', '.join(missing)}")
        return False
    else:
        print("✅ requirements.txt contiene todos los paquetes necesarios")
        return True

def scan_imports_vs_requirements():
    """Escanea imports del proyecto y compara con requirements.txt"""
    print("\n🔎 Escaneo de imports vs requirements:")

    # Mapeo de nombres de import -> paquete pip
    import_to_pip = {
        'sklearn': 'scikit-learn',
        'dateutil': 'python-dateutil',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib',
        'shap': 'shap',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'altair': 'altair',
        'pytest': 'pytest',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
    }

    # Cargar requirements
    req_set = set()
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pkg = line.split('==')[0].lower()
                req_set.add(pkg)

    # Recolectar imports simples (heurística)
    imported = set()
    for root, dirs, files in os.walk('.'):
        # Excluir carpetas no relevantes
        dirs[:] = [d for d in dirs if d not in ['.git', '.venv', 'venv', '__pycache__', '.streamlit', 'models', 'logs', 'data']]
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('import '):
                                # import x, y.z
                                parts = line.replace('import', '', 1).strip().split(',')
                                for p in parts:
                                    top = p.strip().split(' ')[0].split('.')[0]
                                    imported.add(top)
                            elif line.startswith('from '):
                                # from x.y import z
                                mod = line.split(' ')[1].split('.')[0]
                                imported.add(mod)
                except Exception:
                    pass

    # Filtrar a lo que nos interesa (externos conocidos)
    externals = set(import_to_pip.keys())
    imported_externals = sorted(externals.intersection(imported))

    # Convertir a paquetes pip
    needed_pkgs = sorted({import_to_pip[name] for name in imported_externals})

    # Comparar con requirements
    missing = [pkg for pkg in needed_pkgs if pkg not in req_set]

    print(f"   Imports detectados: {', '.join(imported_externals) if imported_externals else '(ninguno)'}")
    if missing:
        print("❌ Faltan en requirements.txt:")
        for m in missing:
            print(f"   - {m}")
        return False
    else:
        print("✅ Todos los imports externos están cubiertos en requirements.txt")
        return True

def check_models():
    """Verifica que los modelos estén entrenados"""
    models_exist = (
        os.path.exists('models/failure_binary_model.joblib') and
        os.path.exists('models/failure_multilabel_models.joblib')
    )
    
    if models_exist:
        print("✅ Modelos entrenados encontrados")
        return True
    else:
        print("❌ Modelos no encontrados - ejecuta: python -m src.ml.train")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 VERIFICACIÓN PRE-DEPLOY PARA STREAMLIT CLOUD")
    print("="*60 + "\n")
    
    # Cambiar al directorio raíz del proyecto
    root_dir = Path(__file__).parent
    os.chdir(root_dir)
    
    all_good = True
    
    # 1. Archivos esenciales
    print("\n📄 Archivos Esenciales:")
    all_good &= check_file_exists('requirements.txt')
    all_good &= check_file_exists('app/streamlit_app.py')
    all_good &= check_file_exists('ai4i2020.csv')
    all_good &= check_file_exists('README.md')
    
    # 2. Archivos de configuración
    print("\n⚙️ Archivos de Configuración:")
    check_file_exists('.gitignore', required=False)
    check_file_exists('.streamlit/config.toml', required=False)
    
    # 3. Directorios necesarios
    print("\n📁 Estructura de Directorios:")
    all_good &= check_directory_exists('src')
    all_good &= check_directory_exists('src/data')
    all_good &= check_directory_exists('src/ml')
    all_good &= check_directory_exists('models')
    check_directory_exists('logs', required=False)
    check_directory_exists('data/additional', required=False)
    
    # 4. Git
    print("\n🔧 Control de Versiones:")
    git_ok = check_git_initialized()
    if not git_ok:
        print("   ⚠️ Ejecuta: git init")
    
    # 5. Requirements
    print("\n📦 Dependencias:")
    all_good &= check_requirements()
    all_good &= scan_imports_vs_requirements()
    
    # 6. Modelos entrenados
    print("\n🤖 Modelos de ML:")
    all_good &= check_models()
    
    # 7. Tamaño de archivos
    print("\n💾 Tamaño de Archivos Grandes:")
    large_files = []
    for root, dirs, files in os.walk('.'):
        # Ignorar .git y __pycache__
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv']]
        
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb > 50:  # Archivos > 50MB
                    large_files.append((filepath, size_mb))
            except:
                pass
    
    if large_files:
        print("⚠️ Archivos grandes encontrados (pueden causar problemas):")
        for filepath, size in large_files:
            print(f"   • {filepath}: {size:.1f} MB")
        print("   Considera usar Git LFS para archivos > 100MB")
    else:
        print("✅ No hay archivos excesivamente grandes")
    
    # Resumen final
    print("\n" + "="*60)
    if all_good:
        print("✅ LISTO PARA DEPLOY")
        print("\nPróximos pasos:")
        print("1. git add .")
        print("2. git commit -m 'Initial commit'")
        print("3. git remote add origin https://github.com/TU_USUARIO/REPO.git")
        print("4. git push -u origin main")
        print("5. Ir a https://share.streamlit.io/ y crear nueva app")
    else:
        print("❌ CORREGE LOS ERRORES ANTES DE HACER DEPLOY")
        print("\nVerifica los elementos marcados con ❌ arriba")
    print("="*60 + "\n")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
