import unittest
import calculator

class TestCalculator(unittest.TestCase):
        ### Test Add ###
    def test_add_pos(self):
        return self.assertEqual(calculator.add(5, 5), 10)

    def test_add_neg(self):
        return self.assertEqual(calculator.add(-5, -5), -10)

    def test_add_mix(self):
        return self.assertEqual(calculator.add(5, -5), 0)

    def test_add_zero(self):
        return self.assertEqual(calculator.add(0, 0), 0)

    def test_add_str(self):
        return self.assertEqual(calculator.add('s', 's'), 'ss')


        ## Test sub ###
    def test_sub_pos(self):
        return self.assertEqual(calculator.sub(5, 5), 0)

    def test_sub_neg(self):
        return self.assertEqual(calculator.sub(-5, -5), 0)

    def test_sub_mix(self):
        return self.assertEqual(calculator.sub(5, -5), 10)

    def test_sub_zero(self):
        return self.assertEqual(calculator.sub(0, 0), 0)

    def test_sub_str_both(self):
        with self.assertRaises(TypeError):
            calculator.sub('s', 's')

    def test_sub_str_x(self):
        with self.assertRaises(TypeError):
            calculator.sub('s', 5)

    def test_sub_str_y(self):
        with self.assertRaises(TypeError):
            calculator.sub(5, 's')
            


        ### Test mul ###
    def test_mul_pos(self):
        return self.assertEqual(calculator.mul(5, 5), 25)

    def test_mul_neg(self):
        return self.assertEqual(calculator.mul(-5, -5), 25)

    def test_mul_mix(self):
        return self.assertEqual(calculator.mul(5, -5), -25)

    def test_mul_zero(self):
        return self.assertEqual(calculator.mul(0, 0), 0)

    def test_mul_int_str(self):
        return self.assertEqual(calculator.mul(5, 's'), 'sssss')

    def test_mul_str_str(self):
        with self.assertRaises(TypeError):
            calculator.mul('s', 's')


        ### Test div ###
    def test_div_pos(self):
        return self.assertEqual(calculator.div(5, 5), 1)

    def test_div_neg(self):
        return self.assertEqual(calculator.div(-5, -5), 1)

    def test_div_mix(self):
        return self.assertEqual(calculator.div(5, -5), -1)

    def test_div_zero_zero(self):
        return self.assertEqual(calculator.div(0, 0), 0)

    def test_div_num_zero(self):
        with self.assertRaises(ZeroDivisionError):
            calculator.div(5, 0)

    def test_div_str(self):
        with self.assertRaises(TypeError):
            calculator.div('s', 's')


if __name__ == '__main__':
    unittest.main(verbosity=2)
