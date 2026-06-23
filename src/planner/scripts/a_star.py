#!/usr/bin/env python3
import rospy
import math
import numpy as np
import os

# Configura o matplotlib para modo "headless" antes de importar o pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from scipy.ndimage import binary_dilation
from tf.transformations import euler_from_quaternion

class NodeAStar:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x  # Linha no grid (Y)
        self.y = y  # Coluna no grid (X)
        self.g = g  # Custo acumulado
        self.h = h  # Heurística
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class TurtlebotAStarStaticNav:
    def __init__(self):
        rospy.init_node('turtlebot3_astar_static', anonymous=True)

        # Metadados do Mapa
        self.grid = None
        self.raw_grid_2d = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = 0
        self.height = 0
        
        # Posição do Robô
        self.pose_x = None
        self.pose_y = None
        self.yaw = 0.0
        
        # Coordenadas do Alvo
        self.goal_real_x = None
        self.goal_real_y = None
        self.new_goal_received = False

        # Estruturas de Debug para o Plot
        self.explored_nodes = set()
        self.final_path_grid = []

        # Configuração da pasta de saída do PNG
        self.pasta_projeto = "/home/user/project"
        self.nome_arquivo_png = "resultado_astar_mapa.png"

        # Publishers e Subscribers
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_publisher = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        self.rate = rospy.Rate(10)

        rospy.loginfo("Aguardando o mapa do map_server...")
        while self.grid is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        rospy.loginfo("Aguardando inicialização da odometria...")
        while (self.pose_x is None or self.pose_y is None) and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        rospy.loginfo(">> TUDO PRONTO! Defina o destino clicando em '2D Nav Goal' no RViz.")

    def map_callback(self, msg):
        if self.grid is not None:
            return
        self.resolution = msg.info.resolution  
        self.origin_x = msg.info.origin.position.x 
        self.origin_y = msg.info.origin.position.y 
        self.width = msg.info.width
        self.height = msg.info.height
        
        raw_data = np.array(msg.data)
        self.raw_grid_2d = raw_data.reshape((self.height, self.width))
        
        # Binarização inicial (Obstáculos e áreas desconhecidas viram 1)
        obstacle_mask = (self.raw_grid_2d == 100) | (self.raw_grid_2d == -1)
        
        # Inflação de segurança via SciPy (Raio de 5 células)
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circular_mask = x**2 + y**2 <= radius**2
        
        rospy.loginfo("Inflando obstáculos via SciPy binary_dilation (C-Space)...")
        inflated_mask = binary_dilation(obstacle_mask, structure=circular_mask)
        
        self.grid = np.where(inflated_mask, 1, 0).tolist()
        rospy.loginfo(f"Grid de Navegação pronto: {self.width}x{self.height}")

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
        rospy.loginfo(f"Novo alvo recebido: X={self.goal_real_x:.2f}, Y={self.goal_real_y:.2f}")
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

    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_publisher.publish(marker)

    def heuristic(self, r1, c1, r2, c2):
        return math.sqrt((r1 - r2)**2 + (c1 - c2)**2)

    def compute_astar(self, start_row, start_col, goal_row, goal_col):
        start_node = NodeAStar(start_row, start_col)
        goal_node = NodeAStar(goal_row, goal_col)
        
        if self.grid[goal_row][goal_col] == 1:
            rospy.logwarn("Destino dentro de obstáculo!")
            return None

        open_list = [start_node]
        self.explored_nodes = set()  # Reseta para o novo plot
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while open_list:
            open_list.sort()
            current_node = open_list.pop(0)
            self.explored_nodes.add((current_node.x, current_node.y)) # Salva para o plot
            
            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                path = []
                while current_node:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                self.final_path_grid = path[::-1]
                return self.final_path_grid
                
            for move in moves:
                nx, ny = current_node.x + move[0], current_node.y + move[1]
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    if self.grid[nx][ny] == 1 or (nx, ny) in self.explored_nodes:
                        continue
                        
                    move_cost = 1.414 if abs(move[0]) == 1 and abs(move[1]) == 1 else 1.0
                    g_cost = current_node.g + move_cost
                    h_cost = self.heuristic(nx, ny, goal_node.x, goal_node.y)
                    neighbor = NodeAStar(nx, ny, g_cost, h_cost, current_node)
                    
                    if any(op.x == nx and op.y == ny and op.g <= g_cost for op in open_list):
                        continue
                    open_list.append(neighbor)
        return None

    def salvar_imagem_debug_astar(self):
        """ Renderiza e salva o mapa com os nós expandidos e o caminho final destacado """
        try:
            caminho_final = os.path.join(self.pasta_projeto, self.nome_arquivo_png)
            plt.figure(figsize=(10, 10))
            
            # Plota o mapa original em tons de cinza
            plt.imshow(self.raw_grid_2d, cmap='gray', origin='lower', alpha=0.5)
            
            # 1. Desenha os nós explorados pelo A* (Nuvem ciano leve de fundo mostrando a busca)
            if len(self.explored_nodes) > 0:
                exp_np = np.array(list(self.explored_nodes))
                plt.scatter(exp_np[:, 1], exp_np[:, 0], color='cyan', s=1, alpha=0.3, label='Espaço de Busca Expandido')
                
            # 2. Desenha o Caminho Final Encontrado (Linha contínua vermelha)
            if len(self.final_path_grid) > 0:
                path_np = np.array(self.final_path_grid)
                plt.plot(path_np[:, 1], path_np[:, 0], color='red', linewidth=2, label='Caminho Ótimo A*')
                
                # Destaca os pontos inicial (verde) e final (azul)
                plt.scatter(path_np[0, 1], path_np[0, 0], color='green', s=100, label='Início', zorder=5)
                plt.scatter(path_np[-1, 1], path_np[-1, 0], color='blue', s=100, label='Meta', zorder=5)

            plt.title("Visualização Ex 1: Expansão de Nós e Rota Ótima A*")
            plt.xlabel("Eixo X (Colunas)")
            plt.ylabel("Eixo Y (Linhas)")
            plt.legend(loc='upper right')
            
            plt.savefig(caminho_final, bbox_inches='tight', dpi=200)
            plt.close()
            rospy.loginfo(f"Sucesso! Imagem do A* salva automaticamente em: {caminho_final}")
        except Exception as e:
            rospy.logerr(f"Erro ao gerar imagem de debug do A*: {str(e)}")

    def navigate_to_waypoints(self, waypoints):
        idx = 0
        while not rospy.is_shutdown() and idx < len(waypoints):
            if self.new_goal_received:
                return False

            target_x, target_y = waypoints[idx]
            distance = math.sqrt((target_x - self.pose_x)**2 + (target_y - self.pose_y)**2)
            angle_to_target = math.atan2(target_y - self.pose_y, target_x - self.pose_x)
            
            twist_msg = Twist()
            if distance < 0.22:
                idx += 1
                continue
                
            angle_error = angle_to_target - self.yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            if abs(angle_error) > 0.4:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 1.3 * angle_error
            else:
                twist_msg.linear.x = min(0.15, 0.4 * distance)
                twist_msg.angular.z = 1.6 * angle_error
                
            self.velocity_publisher.publish(twist_msg)
            self.rate.sleep()
        
        self.velocity_publisher.publish(Twist())
        return True

    def run(self):
        while not rospy.is_shutdown():
            if self.new_goal_received:
                self.new_goal_received = False
                
                start_row, start_col = self.world_to_grid(self.pose_x, self.pose_y)
                goal_row, goal_col = self.world_to_grid(self.goal_real_x, self.goal_real_y)
                
                rospy.loginfo(f"A* Planejando rota: Inicial [{start_row},{start_col}] -> Destino [{goal_row},{goal_col}]")
                grid_path = self.compute_astar(start_row, start_col, goal_row, goal_col)
                
                if grid_path:
                    # Gera a imagem imediatamente após o cálculo do algoritmo
                    self.salvar_imagem_debug_astar()
                    
                    waypoints = [self.grid_to_world(p[0], p[1]) for p in grid_path]
                    waypoints = waypoints[::5] + [waypoints[-1]]
                    
                    rospy.loginfo(f"Rota traçada com sucesso. Executando percurso...")
                    self.navigate_to_waypoints(waypoints)
                else:
                    rospy.logerr("Falha: O algoritmo A* não encontrou uma rota livre de colisões.")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        navigator = TurtlebotAStarStaticNav()
        navigator.run()
    except rospy.ROSInterruptException:
        pass