class UnitNormalizer:
    def __init__(self):
        # ... (映射字典部分保持不变) ...
        self.property_unit_map = {

            "runtime": "seconds",
            "elevation": "meters",
            "height": "meters",
            "mass": "kilograms",
            "weight": "kilograms"
        }
        self.conversion_factors = {
            "seconds": {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400},
            "meters": {"km": 1000, "kilometer": 1000, "cm": 0.01, "foot": 0.3048},
            "kilograms": {"tonne": 1000, "gram": 0.001, "lb": 0.453592, "pound": 0.453592}
        }

    def normalize(self, constraints):
        """
        [修改版] 支持 Constraint 对象操作
        """
        for c in constraints:
            # 使用 getattr 或点号访问对象属性
            prop = c.property_label.lower() if c.property_label else ""
            unit = c.unit
            value = c.value

            if not unit or prop not in self.property_unit_map:
                continue

            target_standard = self.property_unit_map[prop]

            if target_standard in self.conversion_factors:
                factors = self.conversion_factors[target_standard]
                clean_unit = unit.lower().rstrip('s')

                if clean_unit in factors:
                    factor = factors[clean_unit]
                    try:
                        original_val = float(value)
                        new_val = original_val * factor

                        print(
                            f"[UnitNormalizer] Converting {prop}: {original_val} {unit} -> {new_val} {target_standard}")

                        # === 直接修改对象属性 ===
                        c.value = str(new_val)
                        c.unit = None  # 转换完成后清空，避免干扰后续逻辑

                    except ValueError:
                        print(f"[Error] Could not convert value {value} to float.")

        return constraints