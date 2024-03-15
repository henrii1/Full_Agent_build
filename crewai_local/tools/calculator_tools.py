from langchain.tools import tool

class CalculatorTools():

    @tool("Make a calculation")
    def calculate(operation):
        """Useful for perfroming mathematical calculations
           such as summing, substraction, multiplication and
           division operations.

        Args:
            str: string representation of what is to be evaluated
        Returns"
            int: Evaluated output of the input string

        """
        try:
            return eval(operation)
        except SyntaxError:
            return "Error: Invalid syntax in mathematical expression"