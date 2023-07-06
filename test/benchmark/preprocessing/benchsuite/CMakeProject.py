import subprocess
import os

def get_current_path():
    current_path = subprocess.run(["pwd"], stdout=subprocess.PIPE)
    return current_path.stdout.decode('utf-8').replace("\n", "")


class CMakeProject:
    def __init__(self, target, project_path, build_path, cmake_options=[]):
        self.target = target
        self.project_path = project_path
        self.build_path = build_path
        self.cmake_options = cmake_options
        self.executable = None
        self.definitions = {}
        self.print_build_messages = False

    def compile(self, cmake_options=[]):
        old_path = get_current_path()
        if len(cmake_options) == 0:
            cmake_options = self.cmake_options
        cmake_options.append(self.build_definitions_string())
        # self.executable = compile_cmake_project(self.name, self.path, self.build_path, cmake_options)
        
        # navigate to project path
        os.chdir(self.project_path)

        # create build directory
        os.makedirs(self.build_path, exist_ok=True)

        # create command
        command = ["cmake", ".", "-B", self.build_path]
        for option in cmake_options:
            command.append("-D")
            command.append(option)
        
        # command.append(" -- -O3")
        
        if self.print_build_messages:
            print(">>CMake command: " + " ".join(command))
        
        # run cmake
        cmake_configure = subprocess.run(command, stdout=subprocess.PIPE)
        
        if self.print_build_messages:
            print(cmake_configure.stdout.decode('utf-8'))
        
        if cmake_configure.returncode == 0:
            print("> CMake configuration successful")
        
        # build project
        cmake_build = subprocess.run(["cmake", "--build", self.build_path, "--target", self.target], stdout=subprocess.PIPE)
        if self.print_build_messages:
            print(cmake_build.stdout.decode('utf-8'))
        if cmake_build.returncode == 0:
            print("> CMake build successful")
        else:
            print("> CMake build failed")
            os.chdir(old_path)
            exit(1)
            
        self.executable = os.path.abspath(self.build_path + "/" + self.target)
        # os.chdir(old_path)
        
        
    def run(self, pipeOutput=False):
        if self.executable is None:
            print("Executable not set")
            return
        
        # print("Executable: " + self.executable)
        if pipeOutput:
            proc = subprocess.run([self.executable], stdout=subprocess.PIPE)
        else:
            proc = subprocess.run([self.executable])
        # print("Output: " + str(proc.stdout))
        # print("Output: ")
        # for line in proc.stdout.decode('utf-8').splitlines():
        #     print(line)
            
        if proc.returncode != 0:
            if pipeOutput:
                print(proc.stdout.decode('utf-8'))
            print("Error while running executable")
            print("Return code: " + str(proc.returncode))
            exit(1)
        return proc.stdout.decode('utf-8')
        
    # Add a definition to the project
    def add_definition(self, name, value=""):
        self.definitions[name] = value
        
    # Remove a definition from the project
    def remove_definition(self, name):
        self.definitions.pop(name, None)
         
    # Clear all definitions
    def clear_definitions(self):
        self.definitions.clear()
        
    # Build the definitions string
    def build_definitions_string(self):
        definitions_string = "DEFINITIONS="
        for key, value in self.definitions.items():
            definitions_string += key + "=" + value + ";"
        return definitions_string.strip()

    # Switch print build messages on or off
    def switch_print_build_messages(self, value=True):
        self.print_build_messages = value

