# Installer LaTeX sans droits administrateur

## 1. Lancer l'installation

```bash
cd /tmp
cd "$(find . -maxdepth 1 -type d -name 'install-tl-*' | head -n 1)"
perl install-tl --no-interaction --scheme=small --texdir="$HOME/texlive"
```

## 2. Activer LaTeX après l'installation

```bash
export PATH="$HOME/texlive/bin/x86_64-linux:$PATH"
pdflatex --version
```

## 3. Conserver le réglage pour les prochaines connexions

```bash
echo 'export PATH="$HOME/texlive/bin/x86_64-linux:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

## 4. Installer les paquets complémentaires

```bash
tlmgr install collection-latexrecommended collection-latexextra \
  collection-fontsrecommended collection-mathscience \
  collection-pictures collection-binextra lmodern
```

## Compiler un fichier `.tex`

```bash
pdflatex fichier.tex
```
