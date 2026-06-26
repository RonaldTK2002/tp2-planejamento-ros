# Trabalho Prático 2 - Planejamento de Rotas no ROS

Este repositório contém a implementação de três algoritmos clássicos de planejamento de rotas (**A\***, **Voronoi via Brushfire** e **RRT**) aplicados à navegação estática do robô TurtleBot3 em um ambiente simulado no Gazebo com visualização via RViz.

---

## 1. Descrição dos Arquivos de Código

### Scripts de Navegação (`src/planner/scripts/`)
* **`a_star.py`**: Implementa o algoritmo de busca heurística **A\*** padrão sobre uma grade ocupada. Ele infla os obstáculos em um raio de segurança via SciPy e calcula o caminho ótimo (menor custo) entre a posição atual do robô e o alvo selecionado.
* **`brushfire_voronoi.py`**: Executa o algoritmo **Brushfire** para calcular o mapa de distâncias em relação às paredes e extrai um esqueleto do **Diagrama de Voronoi** (linhas centrais das salas). O robô navega utilizando uma variante do A\* que penaliza pesadamente rotas fora das linhas de Voronoi, priorizando o caminho mais central e seguro possível.
* **`rrt.py`**: Implementa a árvore de exploração rápida aleatória (**Rapidly-exploring Random Tree - RRT**). Ele expande caminhos de forma estocástica e incremental até atingir o objetivo com base em um critério de *goal bias* (propensão ao alvo).

### Arquivos de Lançamento e Infraestrutura
* **`src/planner/launch/house.launch`**: Arquivo de inicialização do ROS que carrega o robô no ambiente simulado (Gazebo House), ativa o nó de transformação estática entre os referenciais (`map` para `odom`), inicia o `map_server` para carregar o mapa estático da casa e abre o ambiente visualizador **RViz** pré-configurado.
* **`run_container.bash`**: Script utilizado para **criar e iniciar pela primeira vez** o container Docker baseado em ROS Noetic Desktop Full, configurando o compartilhamento de interface gráfica (X11), aceleração por GPU e mapeamento do workspace.
* **`connect_container.bash`**: Script utilizado para **se reconectar** ao container após ele já ter sido criado. Caso você feche o terminal ou reinicie a máquina, use este comando para abrir uma nova sessão Bash dentro do container existente (`planner_container`) sem perder as modificações.

---

## 2. Instruções de Compilação e Execução

Siga os passos abaixo sequencialmente para configurar o workspace, compilar o projeto e executar as simulações.

### Passo 1: Inicializar o Ambiente Docker
Se for a sua **primeira vez** rodando o projeto nesta máquina, crie o container executando:
```bash
./run_container.bash
```

Caso o container já tenha sido criado anteriormente e você queira apenas voltar a usá-lo (ou abrir um novo terminal dentro dele), execute:

```bash
./connect_container.bash
```
### Passo 2: Instalar as Dependências do Sistema (Dentro do Container)
Atualize o gerenciador de pacotes e instale todas as ferramentas e bibliotecas necessárias (ROS Noetic e SciPy):

```bash
sudo apt update
sudo apt install ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations ros-noetic-map-server python3-scipy
```

### Passo 3: Compilar o Workspace
Navegue até a raiz do seu workspace catkin e execute a compilação:

```bash
catkin_make
source devel/setup.bash
```

### Passo 4: Executar os Programas
Para ver os algoritmos funcionando, você precisará utilizar dois terminais distintos (ambos conectados ao container Docker):

#### Terminal 1: Iniciar o Ambiente Simulador
Navegue até a pasta de launch do projeto e inicie a simulação. Isso abrirá automaticamente o ambiente de testes no Gazebo e a interface gráfica do RViz devidamente configurada.

```bash
cd src/planner/launch
roslaunch house.launch
```

#### Terminal 2: Executar o Script de Navegação
Abra uma nova aba/terminal, reconecte-se ao container usando o `./connect_container.bash`, lembre-se de rodar o source devel/setup.bash para atualizar as variáveis de ambiente do ROS, e execute apenas um dos scripts desejados:

* Para testar o A*:

```bash
python3  src/planner/scripts/a_star.py
```

- Para testar o Brushfire / Voronoi:

```bash
python3  src/planner/scripts/brushfire_voronoi.py
```

+ Para testar o RRT:

```bash
python3  src/planner/scripts/rrt.py
```
## 3. Como Operar a Simulação
Assim que o ambiente do Terminal 1 carregar e o script do Terminal 2 indicar que está pronto (>> TUDO PRONTO!), vá até a janela do RViz.

Na barra de ferramentas superior do RViz, clique no botão 2D Nav Goal (Publish Marker).

Clique e arraste em qualquer ponto livre do mapa para definir a posição e orientação do alvo desejado.

O script correspondente receberá o alvo, calculará a rota gerando um arquivo de imagem de debug em `/home/user/project` e o robô começará a se movimentar autonomamente até o destino.