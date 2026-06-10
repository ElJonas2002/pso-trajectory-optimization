%% Ejecución principal
function robot_pso_comparativo2()
    clearvars; close all; clc;

    % Parámetros del robot planar (2-GDL)
    params.l1 = 0.5; params.l2 = 0.5;         % Longitud de eslabones [m]
    params.m1 = 5.0; params.m2 = 5.0;         % Masa de los eslabones [kg]
    params.I1 = 0.02; params.I2 = 0.015;      % Momento de inercia de los eslabones [kg m^2]
    params.g  = 9.81;                         % Aceleración gravitacional [m/s^2]

    % Puntos inicial y final [rad]
    q0 = [pi/6; pi/3];
    qf = [pi/3; -pi/6];
    
    % Parámetros del PSO
    n_mid = 2;                    % número de puntos intermedios para spline
    dim = 2*n_mid;                % Tamaño de la partícula (PSO)
    bounds = [0, pi/2];            % Límites de ángulos (rad)
    N = 50;                       % Número de partículas
    Tmax = 200;                   % Iteraciones máximas
    T = 4.0; dt = 0.01;           % Tiempo total y dt
    
    objfun = @(x) objective(x, n_mid, params, q0, qf, T, dt);

    [best, val_pso] = pso_global(objfun, bounds, dim, N, Tmax);

    fprintf('\n============== RESULTADO PSO ==============\n');
    best_points = reshape(best, [2, n_mid])';
    disp('Mejores puntos intermedios ([q1 q2] rad):');
    disp(best_points);
    fprintf('Energía final (obj): %.6f J\n', val_pso);

    % Generar trayectoria PSO
    mids_pso = reshape(best, [2, n_mid])';
    points_pso = [q0, mids_pso', qf];
    [t_pso, q_pso, qd_pso, qdd_pso] = generate_trajectory(points_pso, T, dt);
    
    % Calcular energía y torques para PSO
    [energy_pso, peak_torque_pso] = compute_energy_metrics(t_pso, q_pso, qd_pso, qdd_pso, params);

    % Generar trayectoria arbitraria
    [t_arb, q_arb, qd_arb, qdd_arb] = generate_linear_trajectory(q0, qf, T, dt);
    
    % Calcular energía y torques para trayectoria arbitraria
    [energy_arb, peak_torque_arb] = compute_energy_metrics(t_arb, q_arb, qd_arb, qdd_arb, params);
    
    fprintf('\n============== COMPARATIVA ==============\n');
    fprintf('MÉTRICA\t\t\tPSO\t\tARBITRARIA\tMEJORA\n');
    fprintf('Energía (J):\t\t%.2f\t\t%.2f\t\t%.2f%%\n', energy_pso, energy_arb, (energy_arb-energy_pso)/energy_arb*100);
    fprintf('Torque pico (Nm):\t%.2f\t\t%.2f\t\t%.2f%%\n', peak_torque_pso, peak_torque_arb, (peak_torque_arb-peak_torque_pso)/peak_torque_arb*100);

    % Visualización comparativa
    plot_data(t_pso, q_pso, qd_pso, qdd_pso, t_arb, q_arb, qd_arb, qdd_arb, params);
end

%% PSO global
function [best, best_val] = pso_global(f, bounds, dim, N, Tmax)
    omega = 0.6;    % Factor de inercia
    c1 = 1.4;       % Factor cognitivo
    c2 = 1.4;       % Factor social

    % Inicialización
    low = bounds(1); high = bounds(2);
    x = low + (high-low).*rand(N, dim);    % Posiciones de las partículas (N x dim)
    v = zeros(N, dim);                     % Velocidades de las partículas (N x dim)
    pbest = x;                             % Mejor posición personal de cada partícula
    fitness = zeros(N,1);                  
    for i=1:N
        fitness(i) = f(x(i,:));
    end
    [minval, idx] = min(fitness);
    gbest = pbest(idx, :);                 % Mejor posición global
    gbest_val = minval;

    for it = 1:Tmax
        for i = 1:N
            r1 = rand(1,dim); r2 = rand(1,dim);
            v(i,:) = omega.*v(i,:) + c1.*r1.*(pbest(i,:) - x(i,:)) + c2.*r2.*(gbest - x(i,:));
            x(i,:) = x(i,:) + v(i,:);
            x(i,:) = min(max(x(i,:), low), high); % clip para evitar valores fuera de rango

            fxi = f(x(i,:));
            if fxi < fitness(i)
                pbest(i,:) = x(i,:);
                fitness(i) = fxi;
            end
        end

        % Actualizar gbest
        [best_idx_val, best_idx] = min(fitness);
        if best_idx_val < gbest_val
            gbest = pbest(best_idx, :);
            gbest_val = best_idx_val;
        end
        
        % Mostrar progreso cada 50 iteraciones
        if mod(it,50)==0
            fprintf('Iter %d/%d  gbest = %.6f\n', it, Tmax, gbest_val);
        end
    end

    best = gbest;
    best_val = gbest_val;
end

%% Función objetivo
function E = objective(x, n_mid, params, q0, qf, T, dt)
    w_tau = 500;     % Peso del torque
    max_tau = 30;    % Torque máximo [N*m]
    mids = reshape(x, [2, n_mid]);
    points = [q0, mids, qf];       % Pares de ángulos totales

    [t, q, qd, qdd] = generate_trajectory(points, T, dt);

    [total_energy, peak_tau] = compute_energy_metrics(t, q, qd, qdd, params);

    % Penalización de torque (quadratic loss)
    penalty_torque = w_tau * max(0, peak_tau - max_tau)^2;

    % Penalización de velocidad
    max_vel = 2.0; % rad/s (ejemplo)
    penalty_vel = 500 * sum(max(0, abs(qd(:)) - max_vel))^2;

    % Función de costo restringida
    E = total_energy + penalty_torque + penalty_vel;
end

%% Generar trayectoria con spline cúbico natural (por articulación)
function [t, q, qd, qdd] = generate_trajectory(points, T, dt)
    t = 0:dt:T;
    Np = size(points,2);
    % tiempos de los nudos (igual spacing)
    knot_t = linspace(0, T, Np);

    q = zeros(2, numel(t));
    qd = zeros(2, numel(t));
    qdd = zeros(2, numel(t));

    % Para cada articulación construimos un spline cúbico natural:
    for i = 1:2
        yi = points(i, :);
        % csape permite usar 'var' para construir una spline natural
        pp = csape(knot_t, [0, yi, 0], 'clamped');

        % Evaluación del spline (obtener q)
        q(i,:) = ppval(pp, t);

        % Derivadas del spline
        pp1 = fnder(pp, 1);   % primera derivada
        pp2 = fnder(pp, 2);   % segunda derivada

        % Evlauación de las derivadas
        qd(i,:)  = ppval(pp1, t);   % Obtener dot_q
        qdd(i,:) = ppval(pp2, t);   % Obtener ddot_q
    end
end

%% Generar trayectoria lineal en espacio de configuración
function [t, q, qd, qdd] = generate_linear_trajectory(q0, qf, T, dt)
    t = 0:dt:T;
    N = length(t);
    
    q = zeros(2, N);
    qd = zeros(2, N);
    qdd = zeros(2, N);
    
    % Interpolación lineal con suavizado en los extremos
    for i = 1:2
        % Perfil trapezoidal de velocidad (suavizado)
        s = t/T;
        % Suavizar inicio y fin
        s_smooth = 0.5*(1 - cos(pi*s));
        
        q(i,:) = q0(i) + (qf(i) - q0(i)) * s_smooth;
        qd(i,:) = (qf(i) - q0(i)) * (pi/(2*T)) * sin(pi*s);
        qdd(i,:) = (qf(i) - q0(i)) * (pi^2/(2*T^2)) * cos(pi*s);
    end
end

%% Cálculo de métricas de energía
function [total_energy, peak_torque] = compute_energy_metrics(t, q, qd, qdd, params)
    dt = t(2) - t(1);
    n = length(t);
    
    total_energy = 0;
    peak_torque = 0;
    
    for k = 1:n
        tau = torque(q(:,k), qd(:,k), qdd(:,k), params);
        total_energy = total_energy + sum(tau.^2) * dt;
        % power = dot(tau, qd(:,k));
        % total_energy = total_energy + abs(power) * dt;
        
        current_peak = max(abs(tau));
        if current_peak > peak_torque
            peak_torque = current_peak;
        end
    end
end

%% Graficar resultados
function plot_data(t1, q1, qd1, qdd1, t2, q2, qd2, qdd2, params)
    
    % 1. Animación de ambas trayectorias
    animate_robots(q1, q2, params);
    
    % Calcular torques para ambas trayectorias
    tau1 = zeros(2, length(t1));
    tau2 = zeros(2, length(t2));
    
    for k = 1:length(t1)
        tau1(:,k) = torque(q1(:,k), qd1(:,k), qdd1(:,k), params);
    end
    for k = 1:length(t2)
        tau2(:,k) = torque(q2(:,k), qd2(:,k), qdd2(:,k), params);
    end
    % =========================
    % ENERGÍA ACUMULADA
    % =========================
    figure('Name','Accumulated Energy Comparison','NumberTitle','off');
    energy_cum1 = cumsum(sum(tau1.^2), 2) * (t1(2)-t1(1));
    energy_cum2 = cumsum(sum(tau2.^2), 2) * (t2(2)-t2(1));
    plot(t1, energy_cum1, 'b-', 'LineWidth', 2); hold on;
    plot(t2, energy_cum2, 'r--', 'LineWidth', 2);
    xlabel('Tiempo [s]', FontSize= 14, FontName='Computer Modern Roman');
    ylabel('Energía [J]', FontSize= 14, FontName='Computer Modern Roman'); 
    title('Energía Acumulada', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitraria', 'Location', 'best');
    grid on;
    exportgraphics(gcf, 'Energia acumulada.pdf', 'ContentType', 'vector');
    hold off

    % =========================
    % TORQUES
    % =========================
    figure('Name','Joint Torques Comparison','NumberTitle','off');

    ax1 = subplot(1,2,1);
    plot(t1, tau1(1,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, tau2(1,:), 'r--', 'LineWidth', 2);
    ylabel('\tau_1 [Nm]', FontSize= 14, FontName='Computer Modern Roman'); 
    xlabel('Tiempo [s]', FontSize= 14, FontName='Computer Modern Roman');
    title('Torques Articulación 1', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitraria', 'Location', 'best');
    grid on;

    ax2 = subplot(1,2,2);
    plot(t1, tau1(2,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, tau2(2,:), 'r--', 'LineWidth', 2);
    ylabel('\tau_2 [Nm]', FontSize= 14, FontName='Computer Modern Roman'); 
    xlabel('Tiempo [s]', FontSize= 14, FontName='Computer Modern Roman');
    title('Torques Articulación 2', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitraria', 'Location', 'best');
    grid on;

    % Etiquetas (a), (b)
    add_subplot_label(ax1, '(a)');
    add_subplot_label(ax2, '(b)');

    exportgraphics(gcf, 'Torques.pdf', 'ContentType', 'vector');
    hold off

    % =========================
    % POSICIONES ARTICULARES
    % =========================
    figure('Name','Joint Positions Comparison','NumberTitle','off');
    
    % Articulación 1
    ax3 = subplot(1,2,1);
    plot(t1, q1(1,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, q2(1,:), 'r--', 'LineWidth', 2);
    xlabel('Time [s]', FontSize= 14, FontName='Computer Modern Roman'); 
    ylabel('q_1 [rad]', FontSize= 14, FontName='Computer Modern Roman');
    title('Joint 1 Position', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitrary', 'Location', 'best');
    grid on;
    
    % Articulación 2
    ax4 = subplot(1,2,2);
    plot(t1, q1(2,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, q2(2,:), 'r--', 'LineWidth', 2);
    xlabel('Time [s]', FontSize= 14, FontName='Computer Modern Roman');
    ylabel('q_2 [rad]', FontSize= 14, FontName='Computer Modern Roman');
    title('Joint 2 Position', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitrary', 'Location', 'best');
    grid on;

    % Etiquetas (a), (b)
    add_subplot_label(ax3, '(a)');
    add_subplot_label(ax4, '(b)');

    exportgraphics(gcf, 'Posiciones.pdf', 'ContentType', 'vector');
    hold off

    % =========================
    % VELOCIDADES
    % =========================
    figure('Name','Joint Velocities Comparison','NumberTitle','off');

    ax5 = subplot(1,2,1);
    plot(t1, qd1(1,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, qd2(1,:), 'r--', 'LineWidth', 2);
    xlabel('Tiempo [s]', FontSize= 14, FontName='Computer Modern Roman'); 
    ylabel('dq_1/dt [rad/s]', FontSize= 14, FontName='Computer Modern Roman'); 
    title('Velocidad Articulación 1', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitraria', 'Location', 'best');
    grid on;

    ax6 = subplot(1,2,2);
    plot(t1, qd1(2,:), 'b-', 'LineWidth', 2); hold on;
    plot(t2, qd2(2,:), 'r--', 'LineWidth', 2);
    xlabel('Tiempo [s]', FontSize= 14, FontName='Computer Modern Roman');
    ylabel('dq_2/dt [rad/s]', FontSize= 14, FontName='Computer Modern Roman'); 
    title('Velocidad Articulación 2', FontSize= 14, FontName='Computer Modern Roman');
    legend('PSO', 'Arbitraria', 'Location', 'best');
    grid on;

    % Etiquetas (a), (b)
    add_subplot_label(ax5, '(a)');
    add_subplot_label(ax6, '(b)');

    exportgraphics(gcf, 'Velocidades.pdf', 'ContentType', 'vector');
    hold off
end

% ==========================================
% FUNCIONES AUXILIARES PARA ETIQUETAS
% ==========================================
function add_subplot_label(ax, label)
    pos = ax.Position;
    annotation('textbox', ...
        [pos(1), pos(2)-0.1, pos(3), 0.05], ...
        'String', label, ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold',        ...
        'FontSize', 14, ...
        'FontName', 'Computer Modern Roman');
end

function add_subplot_label_anim(ax, label)
    pos = ax.Position;
    annotation('textbox', ...
        [pos(1), pos(2)-0.03, pos(3), 0.05], ...
        'String', label, ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold', ...
        'FontSize', 14, ...
        'FontName', 'Computer Modern Roman');
end

%% Animación comparativa
function animate_robots(q1, q2, params)
    figure('Name','Trajectories Comparison','NumberTitle','off');
    
    % Cinemática directa
    fk = @(qv) [ 0, params.l1*cos(qv(1)), params.l1*cos(qv(1)) + params.l2*cos(qv(1)+qv(2)); ...
                 0, params.l1*sin(qv(1)), params.l1*sin(qv(1)) + params.l2*sin(qv(1)+qv(2)) ];

    % Calcular trayectorias cartesianas
    trajX1 = zeros(1, size(q1,2)); trajY1 = zeros(1, size(q1,2));
    trajX2 = zeros(1, size(q2,2)); trajY2 = zeros(1, size(q2,2));
    
    for k = 1:size(q1,2)
        P = fk(q1(:,k));
        trajX1(k) = P(1,3); trajY1(k) = P(2,3);
    end
    for k = 1:size(q2,2)
        P = fk(q2(:,k));
        trajX2(k) = P(1,3); trajY2(k) = P(2,3);
    end
    Pi = [trajX1(:,1), trajY1(:,1)];
    Pf = [trajX1(:,end), trajY1(:,end)];
    
    % Subplot 1: Trayectoria PSO
    subplot(1,2,1);
    ax1 = gca;
    hold on; axis equal; grid on;
    axis([-1.1 1.1 -1.1 1.1]);
    title('Trayectoria Generada con PSO', FontSize= 14, FontName='Computer Modern Roman');
    xlabel('x [m]', FontSize= 14, FontName='Computer Modern Roman');
    ylabel('y [m]', FontSize= 14, FontName='Computer Modern Roman');
    plot(trajX1, trajY1, 'b--', 'LineWidth', 1, "HandleVisibility", "off");
    h_link1 = plot([0,0], [0,0], 'ro-', 'LineWidth', 3, 'MarkerSize', 6, "HandleVisibility", "off");
    scatter(Pi(1), Pi(2), 80, "magenta", "filled", "DisplayName", "Punto Inicial \theta_0")
    scatter(Pf(1), Pf(2), 80, 'g', 'filled', 'DisplayName', 'Punto Final \theta_f');
    legend('Location','northwest')
    
    % Subplot 2: Trayectoria Arbitraria
    subplot(1,2,2);
    ax2 = gca;
    hold on; axis equal; grid on;
    axis([-1.1 1.1 -1.1 1.1]);
    title('Trayectoria Arbitraria', FontSize= 14, FontName='Computer Modern Roman');
    xlabel('x [m]', FontSize= 14, FontName='Computer Modern Roman');
    ylabel('y [m]', FontSize= 14, FontName='Computer Modern Roman');
    plot(trajX2, trajY2, 'r--', 'LineWidth', 1, "HandleVisibility", "off");
    h_link2 = plot([0,0], [0,0], 'ro-', 'LineWidth', 3, 'MarkerSize', 6, "HandleVisibility", "off");
    scatter(Pi(1), Pi(2), 80, "magenta", "filled", "DisplayName", "Punto Inicial \theta_0")
    scatter(Pf(1), Pf(2), 80, 'g', 'filled', 'DisplayName', 'Punto Final \theta_f');
    legend('Location',"northwest")

    add_subplot_label_anim(ax1, '(a)')
    add_subplot_label_anim(ax2, '(b)')

    % Animación simultánea
    for k = 1:min(size(q1,2), size(q2,2))
        % Robot PSO
        subplot(1,2,1);
        P1 = fk(q1(:,k));
        set(h_link1, 'XData', P1(1,:), 'YData', P1(2,:));
        
        % Robot Arbitrario
        subplot(1,2,2);
        P2 = fk(q2(:,k));
        set(h_link2, 'XData', P2(1,:), 'YData', P2(2,:));
        
        drawnow;
        pause(0.01);
    end
    exportgraphics(gcf, 'Trayectorias.pdf', 'ContentType', 'vector')
    close
end

%% Torques de los motores
function tau = torque(q, qd, qdd, params)
    M = Mmat(q, params);
    C = Cvec(q, qd, params);
    G = Gvec(q, params);
    tau = M*qdd + C + G;
end

%% Matriz de masa
function M = Mmat(q, params)
    th2 = q(2);
    c2 = cos(th2);
    l1 = params.l1; l2 = params.l2;
    m1 = params.m1; m2 = params.m2;
    I1 = params.I1; I2 = params.I2;

    M11 = I1 + I2 + m1*(l1^2)/4 + m2*(l1^2 + l2^2/4 + l1*l2*c2);
    M12 = I2 + m2*(l2^2/4 + 0.5*l1*l2*c2);
    M21 = M12;
    M22 = I2 + m2*(l2^2/4);
    M = [M11, M12; M21, M22];
end

%% Vector de Coriolis / Centrífugo
function Cvec = Cvec(q, qd, params)
    th2 = q(2);
    d1 = qd(1); d2 = qd(2);

    l1 = params.l1; l2 = params.l2;
    m2 = params.m2;

    h = m2*(l1*l2/2)*sin(th2);

    C1 = -h*(2*d1*d2 + d2^2);
    C2 =  h*(d1^2);

    Cvec = [C1; C2];
end

%% Vector de gravedad
function G = Gvec(q, params)
    % Asume centros de masa en l1/2 y l2/2
    th1 = q(1); th2 = q(2);

    l1 = params.l1; l2 = params.l2;
    m1 = params.m1; m2 = params.m2;
    g  = params.g;

    % G1: contribución de m1 en l1/2 y de m2 (tanto por l1 como por l2/2)
    G1 = m1*(l1/2)*g*cos(th1) + m2*(l1*g*cos(th1) + (l2/2)*g*cos(th1+th2));
    % G2: solo la parte debida al segundo eslabón (centro en l2/2)
    G2 = m2*(l2/2)*g*cos(th1+th2);

    G = [G1; G2];
end
