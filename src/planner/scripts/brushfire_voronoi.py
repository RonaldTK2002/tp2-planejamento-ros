#!/usr/bin/env python3
import rospy
import math
import numpy as np
import os

# Configura o matplotlib para modo "headless" antes de importar o pyplot
# Isso evita que o ROS quebre tentando abrir janelas de interface gráfica em background
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
from collections import deque

class NodeAStar:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x # Linha (Y)
        self.y = y # Coluna (X)
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class TurtlebotVoronoiNav:
    def __init__(self):
        rospy.init_node('turtlebot3_voronoi_brushfire', anonymous=True)

        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = 0
        self.height = 0
        
        # Matrizes do Brushfire e Voronoi
        self.distance_map = None  
        self.voronoi_grid = None  
        self.voronoi_matrix = None
        
        self.pose_x = None
        self.pose_y = None
        self.yaw = 0.0
        
        self.goal_real_x = None
        self.goal_real_y = None
        self.new_goal_received = False

        # Configuração do caminho de saída do PNG
        # TODO: Altere esta pasta se quiser salvar em outro lugar
        self.pasta_projeto = "/home/user/project"
        self.nome_arquivo_png = "resultado_brushfire_ros.png"

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_publisher = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        self.rate = rospy.Rate(10)

        rospy.loginfo("Aguardando mapa do map_server...")
        while self.voronoi_grid is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo(">> EXERCÍCIO 2 PRONTO! Escolha o alvo pelo RViz.")

    def map_callback(self, msg):
        if self.voronoi_grid is not None:
            return
        self.resolution = msg.info.resolution  
        self.origin_x = msg.info.origin.position.x 
        self.origin_y = msg.info.origin.position.y 
        self.width = msg.info.width
        self.height = msg.info.height
        
        raw_data = np.array(msg.data)
        grid_2d = raw_data.reshape((self.height, self.width))
        
        # 1. Executa o algoritmo Brushfire e Voronoi nas matrizes
        self.compute_brushfire_and_voronoi(grid_2d)
        
        # 2. Gera e salva o PNG automaticamente no mesmo nó
        self.salvar_imagem_debug()

    def compute_brushfire_and_voronoi(self, grid_2d):
            rospy.loginfo("Iniciando Algoritmo Brushfire no nó...")
            
            # 1. COMPUTAÇÃO DO BRUSHFIRE (MAPA DE DISTÂNCIAS)
            self.distance_map = np.full((self.height, self.width), np.inf)
            queue = deque()
            
            walls = (grid_2d == 100) | (grid_2d == -1)
            self.distance_map[walls] = 0
            
            positions = np.argwhere(walls)
            for r, c in positions:
                queue.append((r, c))
                
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while queue:
                r, c = queue.popleft()
                current_dist = self.distance_map[r, c]
                
                for dr, dc in moves:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if self.distance_map[nr, nc] == np.inf:
                            self.distance_map[nr, nc] = current_dist + 1
                            queue.append((nr, nc))
                            
            rospy.loginfo("Brushfire concluído! Extraindo esqueleto fino de Voronoi...")
            
            # 2. EXTRAÇÃO DE CRISTAS FINAS (MÁXIMOS LOCAIS ESTRITOS)
            self.voronoi_grid = np.zeros((self.height, self.width), dtype=int)
            min_clearance = 5  # Aumentado levemente para afastar mais das quinas
            
            for r in range(1, self.height - 1):
                for c in range(1, self.width - 1):
                    val = self.distance_map[r, c]
                    
                    if val < min_clearance:
                        continue
                    
                    # Vizinhos diretos (4-vizinhos)
                    v_cima  = self.distance_map[r+1, c]
                    v_baixo = self.distance_map[r-1, c]
                    v_dir   = self.distance_map[r, c+1]
                    v_esq   = self.distance_map[r, c-1]
                    
                    # Para ser uma crista central real, a célula deve ser um máximo local estrito 
                    # em pelo menos uma das direções (vertical ou horizontal), sem margem de tolerância.
                    # Isso impede que o centro plano de salas grandes seja todo preenchido.
                    is_max_vertical   = (val > v_cima and val >= v_baixo) or (val >= v_cima and val > v_baixo)
                    is_max_horizontal = (val > v_dir and val >= v_esq) or (val >= v_dir and val > v_esq)
                    
                    if is_max_vertical or is_max_horizontal:
                        self.voronoi_grid[r, c] = 1

            # Converte a matriz final para lista Python
            self.voronoi_matrix = self.voronoi_grid.tolist()
            rospy.loginfo("Diagrama de Voronoi Calibrado gerado com sucesso!")
    def salvar_imagem_debug(self):
        """ Renderiza o mapa de distância e a linha de Voronoi e exporta para PNG """
        try:
            rospy.loginfo("Gerando imagem de visualização do Brushfire...")
            caminho_final = os.path.join(self.pasta_projeto, self.nome_arquivo_png)
            
            plt.figure(figsize=(10, 10))
            
            # Desenha o campo de distâncias do Brushfire de fundo (Gradiente)
            plt.imshow(self.distance_map, cmap='viridis', origin='lower')
            plt.colorbar(label='Distância até a parede (células)')
            
            # Extrai as coordenadas onde o Voronoi foi ativado para sobrepor pontos vermelhos
            v_positions = np.argwhere(self.voronoi_grid == 1)
            if len(v_positions) > 0:
                v_y, v_x = v_positions[:, 0], v_positions[:, 1]
                plt.scatter(v_x, v_y, c='red', s=1, label='Caminhos de Voronoi', alpha=0.6)
            
            plt.title("Visualização Integrada ROS: Campo de Distâncias & Voronoi")
            plt.xlabel("Colunas (Eixo X)")
            plt.ylabel("Linhas (Eixo Y)")
            plt.legend(loc='upper right')
            
            plt.savefig(caminho_final, bbox_inches='tight', dpi=200)
            plt.close() # Limpa a memória do pyplot
            rospy.loginfo(f"Sucesso! Imagem de debug salva automaticamente em: {caminho_final}")
        except Exception as e:
            rospy.logerr(f"Erro ao gerar a imagem do Brushfire: {str(e)}")

    def odom_callback(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.yaw) = euler_from_quaternion(orientation_list)

    def goal_callback(self, msg):
        self.goal_real_x = msg.pose.position.x
        self.goal_real_y = msg.pose.position.y
        self.new_goal_received = True
        self.publish_goal_marker(self.goal_real_x, self.goal_real_y)

    def world_to_grid(self, real_x, real_y):
        col = int((real_x - self.origin_x) / self.resolution)
        row = int((real_y - self.origin_y) / self.resolution)
        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))
        return row, col

    def grid_to_world(self, row, col):
        real_x = (col * self.resolution) + self.origin_x + (self.resolution / 2.0)
        real_y = (row * self.resolution) + self.origin_y + (self.resolution / 2.0)
        return real_x, real_y

    def heuristic(self, r1, c1, r2, c2):
        return math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

    def compute_astar_voronoi(self, start_row, start_col, goal_row, goal_col):
        start_node = NodeAStar(start_row, start_col)
        goal_node = NodeAStar(goal_row, goal_col)
        
        if self.distance_map[goal_row, goal_col] < 3:
            rospy.logwarn("Meta muito colada na parede física!")
            return None

        open_list = [start_node]
        closed_list = set()
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while open_list:
            open_list.sort()
            current_node = open_list.pop(0)
            closed_list.add((current_node.x, current_node.y))
            
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]
                
            for move in moves:
                nx, ny = current_node.x + move[0], current_node.y + move[1]
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    if self.distance_map[nx, ny] == 0 or (nx, ny) in closed_list:
                        continue
                    
                    move_cost = 1.414 if abs(move[0]) == 1 and abs(move[1]) == 1 else 1.0
                    
                    # Penalidade pesada para caminhos fora de Voronoi
                    if self.voronoi_matrix[nx][ny] == 0:
                        voronoi_penalty = 15.0 / (self.distance_map[nx, ny] + 0.1)
                    else:
                        voronoi_penalty = 0.0 
                        
                    g_cost = current_node.g + move_cost + voronoi_penalty
                    h_cost = self.heuristic(nx, ny, goal_node.x, goal_node.y)
                    
                    neighbor = NodeAStar(nx, ny, g_cost, h_cost, current_node)
                    
                    if any(op.x == nx and op.y == ny and op.g <= g_cost for op in open_list):
                        continue
                    open_list.append(neighbor)
        return None

    def navigate_to_waypoints(self, waypoints):
        idx = 0
        while not rospy.is_shutdown() and idx < len(waypoints):
            if self.new_goal_received:
                return False

            target_x, target_y = waypoints[idx]
            distance = math.sqrt((target_x - self.pose_x)**2 + (target_y - self.pose_y)**2)
            angle_to_target = math.atan2(target_y - self.pose_y, target_x - self.pose_x)
            
            twist_msg = Twist()
            if distance < 0.25: 
                idx += 1
                continue
                
            angle_error = angle_to_target - self.yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            if abs(angle_error) > 0.4:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 1.3 * angle_error
            else:
                twist_msg.linear.x = min(0.14, 0.4 * distance)
                twist_msg.angular.z = 1.6 * angle_error
                
            self.velocity_publisher.publish(twist_msg)
            self.rate.sleep()
        
        self.velocity_publisher.publish(Twist())
        return True

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.15
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  
        marker.color.a = 1.0
        self.marker_publisher.publish(marker)

    def run(self):
        while not rospy.is_shutdown():
            if self.new_goal_received:
                self.new_goal_received = False
                
                start_row, start_col = self.world_to_grid(self.pose_x, self.pose_y)
                goal_row, goal_col = self.world_to_grid(self.goal_real_x, self.goal_real_y)
                
                rospy.loginfo("Calculando caminho pelas linhas centrais de Voronoi...")
                grid_path = self.compute_astar_voronoi(start_row, start_col, goal_row, goal_col)
                
                if grid_path:
                    waypoints = [self.grid_to_world(p[0], p[1]) for p in grid_path]
                    waypoints = waypoints[::5] + [waypoints[-1]]
                    
                    rospy.loginfo("Iniciando percurso centralizado e seguro.")
                    self.navigate_to_waypoints(waypoints)
                else:
                    rospy.logerr("Falha ao traçar rota pelo Diagrama de Voronoi.")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        navigator = TurtlebotVoronoiNav()
        navigator.run()
    except rospy.ROSInterruptException:
        pass