# 电机偏差补偿模块
# 用于对左右轮速度或里程计进行校正，提高运动精度

from config import settings

def apply_motor_deviation_correction(linear_velocity, angular_velocity):
    """
    对输入的线速度和角速度进行电机偏差补偿，返回修正后的速度。
    linear_velocity: 机器人线速度（m/s）
    angular_velocity: 机器人角速度（rad/s）
    返回: (left_wheel, right_wheel) 校正后的左右轮速度（m/s）
    """
    # 获取校正系数
    k_left = getattr(settings, 'MOTOR_LEFT_CORRECTION', 1.0)
    k_right = getattr(settings, 'MOTOR_RIGHT_CORRECTION', 1.0)
    wheel_base = getattr(settings, 'ROBOT_LENGTH', 0.3)  # 轮距，单位米

    # 由线速度和角速度反解左右轮速度
    v_l = linear_velocity - 0.5 * wheel_base * angular_velocity
    v_r = linear_velocity + 0.5 * wheel_base * angular_velocity

    # 应用校正系数
    v_l_corr = v_l * k_left
    v_r_corr = v_r * k_right

    return v_l_corr, v_r_corr


def correct_odometry(left_odom, right_odom):
    """
    对左右轮里程计读数进行偏差补偿。
    left_odom: 左轮原始里程计
    right_odom: 右轮原始里程计
    返回: (left_corr, right_corr)
    """
    k_left = getattr(settings, 'MOTOR_LEFT_CORRECTION', 1.0)
    k_right = getattr(settings, 'MOTOR_RIGHT_CORRECTION', 1.0)
    return left_odom * k_left, right_odom * k_right