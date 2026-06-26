#!/usr/bin/env python3
import rospy
import math
import numpy as np
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from scipy.ndimage import binary_dilation
from tf.transformations import euler_from_quaternion

class RRTNode:
    def __init__(self, r, c):
        self.r = r  # Linha no grid (Y)
        self.c = c  # Coluna no grid (X)
        self.parent = None

class TurtlebotRRTNav:
    def __init__(self):
        rospy.init_node('turtlebot3_rrt_nav', anonymous=True)

        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.width = 0
        self.height = 0
        
        # Grid binário de obstáculos (0 livre, 1 parede/inflado)
        self.grid = None  
        self.raw_grid_2d = None
        
        self.pose_x = None
        self.pose_y = None
        self.yaw = 0.0
        
        self.goal_real_x = None
        self.goal_real_y = None
        self.new_goal_received = False

        # Estrutura para salvar a última árvore gerada para o plot
        self.all_edges = []
        self.final_path_grid = []

        # Configuração da pasta de saída do PNG
        self.pasta_projeto = "/home/user/project"
        self.nome_arquivo_png = "resultado_rrt_arvore.png"

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_publisher = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        self.rate = rospy.Rate(10)

        rospy.loginfo("Aguardando mapa...")
        while self.grid is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo(">> EXERCÍCIO 3 PRONTO! Defina o alvo no RViz usando RRT.")

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
        
        # Binarização inicial (Obstáculos e áreas desconhecidas viram True/1)
        obstacle_mask = (self.raw_grid_2d == 100) | (self.raw_grid_2d == -1)
        
        # Inflação de segurança idêntica à do A* via SciPy (Raio de 5 células)
        radius = 5
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circular_mask = x**2 + y**2 <= radius**2
        
        rospy.loginfo("Inflando obstáculos do RRT via SciPy binary_dilation...")
        inflated_mask = binary_dilation(obstacle_mask, structure=circular_mask)
        
        # Salva como array do NumPy (0 livre, 1 obstáculo) para manter compatibilidade com o check_collision
        self.grid = np.where(inflated_mask, 1, 0)
        rospy.loginfo(f"Grid do RRT inicializado com C-Space do SciPy: {self.width}x{self.height}")

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

    def check_collision(self, r1, c1, r2, c2):
        """ Checa por amostragem de passos se a linha entre dois nós corta uma parede """
        steps = int(math.sqrt((r1 - r2)**2 + (c1 - c2)**2)) * 2
        if steps == 0:
            return False
        for i in range(steps + 1):
            t = i / steps
            r = int(r1 + t * (r2 - r1))
            c = int(c1 + t * (c2 - c1))
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.grid[r, c] == 1:
                    return True # Colisão detectada
            else:
                return True
        return False

    def compute_rrt(self, start_row, start_col, goal_row, goal_col):
        rospy.loginfo("Planejando rota com RRT...")
        start_node = RRTNode(start_row, start_col)
        
        if self.grid[goal_row, goal_col] == 1:
            rospy.logwarn("Alvo inválido selecionado (está dentro de um obstáculo)!")
            return None

        nodes = [start_node]
        self.all_edges = []  # Reseta para o novo plot
        
        max_iter = 5000
        step_size = 6  # Tamanho do passo máximo de extensão (em células do grid)
        goal_bias = 0.15  # 15% de chance de puxar o sorteio direto para o objetivo final
        
        for _ in range(max_iter):
            # Goal Bias: decide se sorteia um ponto qualquer ou mira direto no objetivo
            if random.random() < goal_bias:
                rand_r, rand_c = goal_row, goal_col
            else:
                rand_r = random.randint(0, self.height - 1)
                rand_c = random.randint(0, self.width - 1)
                
            # Encontra o nó da árvore mais perto do ponto sorteado
            nearest_node = nodes[0]
            min_dist = math.sqrt((nearest_node.r - rand_r)**2 + (nearest_node.c - rand_c)**2)
            for node in nodes:
                d = math.sqrt((node.r - rand_r)**2 + (node.c - rand_c)**2)
                if d < min_dist:
                    min_dist = d
                    nearest_node = node
                    
            # Calcula a direção e estende a árvore em direção ao ponto sorteado com o step_size
            theta = math.atan2(rand_r - nearest_node.r, rand_c - nearest_node.c)
            new_r = int(nearest_node.r + step_size * math.sin(theta))
            new_c = int(nearest_node.c + step_size * math.cos(theta))
            
            # Limita as bordas do mapa
            new_r = max(0, min(new_r, self.height - 1))
            new_c = max(0, min(new_c, self.width - 1))
            
            # Valida o novo segmento
            if not self.check_collision(nearest_node.r, nearest_node.c, new_r, new_c):
                new_node = RRTNode(new_r, new_c)
                new_node.parent = nearest_node
                nodes.append(new_node)
                
                # Guarda a linha para visualizarmos a árvore depois
                self.all_edges.append(((nearest_node.c, nearest_node.r), (new_c, new_r)))
                
                # Checa se o novo nó chegou perto o suficiente do destino final
                dist_to_goal = math.sqrt((new_r - goal_row)**2 + (new_c - goal_col)**2)
                if dist_to_goal <= step_size:
                    if not self.check_collision(new_r, new_c, goal_row, goal_col):
                        goal_final_node = RRTNode(goal_row, goal_col)
                        goal_final_node.parent = new_node
                        nodes.append(goal_final_node)
                        self.all_edges.append(((new_node.c, new_r), (goal_col, goal_row)))
                        
                        # Reconstrói e retorna o caminho
                        path = []
                        curr = goal_final_node
                        while curr:
                            path.append((curr.r, curr.c))
                            curr = curr.parent
                        self.final_path_grid = path[::-1]
                        return self.final_path_grid
                        
        return None

    def salvar_imagem_debug_rrt(self):
        """ Renderiza e salva a árvore de nós e o caminho destacado por cima do mapa """
        try:
            caminho_final = os.path.join(self.pasta_projeto, self.nome_arquivo_png)
            plt.figure(figsize=(10, 10))
            
            # Plota o mapa original em tons de cinza
            plt.imshow(self.raw_grid_2d, cmap='gray', origin='lower', alpha=0.6)
            
            # 1. Desenha todas as arestas da Árvore RRT gerada (Linhas cinzas/finas)
            for edge in self.all_edges:
                p1, p2 = edge
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', linewidth=0.5, alpha=0.5)
                
            # 2. Destaca o Caminho Escolhido Final (Linha Vermelha Grossa)
            if len(self.final_path_grid) > 0:
                path_np = np.array(self.final_path_grid)
                plt.plot(path_np[:, 1], path_np[:, 0], color='red', linewidth=2.5, label='Caminho Selecionado RRT')
                
                # Plota Início (Verde) e Fim (Azul)
                plt.scatter(path_np[0, 1], path_np[0, 0], color='green', s=100, label='Início', zorder=5)
                plt.scatter(path_np[-1, 1], path_np[-1, 0], color='blue', s=100, label='Alvo', zorder=5)

            plt.title("Visualização Ex 3: Árvore de Exploração RRT e Rota Final")
            plt.xlabel("Eixo X (Colunas)")
            plt.ylabel("Eixo Y (Linhas)")
            plt.legend(loc='upper right')
            
            plt.savefig(caminho_final, bbox_inches='tight', dpi=200)
            plt.close()
            rospy.loginfo(f"Sucesso! Imagem da árvore salva em: {caminho_final}")
        except Exception as e:
            rospy.logerr(f"Erro ao desenhar imagem do RRT: {str(e)}")

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
        marker.color.g = 1.0  # Verde para o RRT
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_publisher.publish(marker)

    def run(self):
        while not rospy.is_shutdown():
            if self.new_goal_received:
                self.new_goal_received = False
                
                start_row, start_col = self.world_to_grid(self.pose_x, self.pose_y)
                goal_row, goal_col = self.world_to_grid(self.goal_real_x, self.goal_real_y)
                
                grid_path = self.compute_rrt(start_row, start_col, goal_row, goal_col)
                
                if grid_path:
                    # Salva a árvore e destaca a rota na imagem PNG imediatamente
                    self.salvar_imagem_debug_rrt()
                    
                    waypoints = [self.grid_to_world(p[0], p[1]) for p in grid_path]
                    # Como o RRT nativamente já gera nós espaçados, podemos enviar direto ou suavizar levemente
                    waypoints = waypoints[::2] + [waypoints[-1]]
                    
                    rospy.loginfo("Rota calculada pelo RRT. Executando navegação...")
                    self.navigate_to_waypoints(waypoints)
                else:
                    rospy.logerr("O RRT atingiu o limite de iterações e não conseguiu conectar os pontos.")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        navigator = TurtlebotRRTNav()
        navigator.run()
    except rospy.ROSInterruptException:
        pass