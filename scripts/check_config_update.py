#!/usr/bin/env python

# Author: Aza Tulepbergenov
# Author: Sarah Willer
# Date: August 2016, revised December 2016-January 2017
#
##############################################################################
# If this file is placed somewhere other than the directory that contains the
# git repository, update the filepath in .git/hooks/pre-commit accordingly.
##############################################################################
#
# check_config_update.py is called by the bash script .git/hooks/pre-commit on
# any new or changed files containing the search phrase 'addParam'.
#
# For each file on which it is called, check_config_update.py looks for added
# configuration options and prints them under the file name in the commit
# status message with a request to update new options to Configuration.txt.
# 
# Script was tested on variety of scenarios, including addParam calls spanning multiple lines,
# commenting/uncommenting addParamc calls, moving files around, deleting them etc. 
# Regular expressions have been used as the main pattern recognition method.
# 
# If you encounter any bugs, please contact Aza Tulepbergenov (@atulep)
#
# Resources referenced in writing this script:
# https://www.atlassian.com/git/tutorials/git-hooks/local-hooks
# http://stackoverflow.com/questions/4940032/search-for-string-in-txt-file-python
# https://pypi.python.org/pypi/git-pre-commit-hook
# http://stackoverflow.com/questions/24968112/searching-files-with-pre-commit
# http://unix.stackexchange.com/questions/24140/return-only-the-portion-of-a-line-after-a-matching-pattern
# http://stackoverflow.com/questions/8369219/how-do-i-read-a-text-file-into-a-string-variable-in-python
# http://pyregex.com 
#############################################################################

import sys
import re
import os
import subprocess

# assume that config file will be inside of the ./themes/ subdirectory, where
# current directory is the root directory of the repo.
PATH_TO_CONFIG_FILE = os.getcwd() + '/themes/Configuration.txt'

# constants used to delimit sections in the Confifguration.txt
BORDER_BEGIN = 'BEGIN '
BORDER_END = 'END '
GENERAL = 'GENERAL'

# 'mapping' exists because Jacobi section in the Configuration.txt blended both Jacobi1D and Jacobi2D together, but
# there are two folders for each Jacobi in the project directory.
mapping = {'Jacobi1D': 'JACOBI',
           'Jacobi2D': 'JACOBI',
           'MiniFluxDiv': 'MINIFLUX',
           'VIBE': 'VIBE',
           'BlurRoberts': 'BLURROBERTS',
           'common': 'COMMON'}

# if enabled, prints useful info in various places inside the code.
DEBUG = False


def main():
    """
    The .git/hooks/pre_commit bash script will pass in the name of the file that contain newly added
    'addParam' methods. This function will check if the newly added 'addParam's are present in the Configuration.txt.
    If they don't, then script will exit with status 1.
    """
    # the file name will be a relative path from the root git directory
    file_name = sys.argv[1]

    # git should ignore the changes done in this file
    if file_name in __file__:
        return

    # file_name = os.getcwd() + '../themes/Jacobi2D/Jacobi2D-Diamond-OMP.cpp'
    is_deleted = check_deleted_files(file_name)
    # git should ignore the deleted files
    if is_deleted:
        return 
    
    # names from specific and general sections of the config file
    names_from_config_file = pre_process(file_name)

    param_map = {}
    
    # since we're only interested in presence of paramaters and not their occurences, we use a set.
    # NOTE: using a list will contain duplicates even if the elements in the files are unique. This is a 
    # minor bug, which I wish to solve in future.
    names_from_comitted_file = set()
    
    # compiling regular expression outside of a loop is more efficient
    # will match '.addParam(...)'.
    regex = re.compile(r'[?=.](addParam[^;]+)')

    # memory efficient way of parsing the file.
    with open(file_name, 'r') as current_file:
        multi_line_comment = False
        st = False # flag is set True when there is a multi-line addParam call 
        # iterate through lines
        for line in current_file:
            # start skipping the multi-line comments
            # will set the flag to true, but not immediately continue.
            # this is done to account for case when comment is of form /**/
            if '/*' in line:
                multi_line_comment = True
            # stop skipping the multi-line comments
            if '*/' in line:
                multi_line_comment = False
                continue 
            # skip single line comments
            if '//' in line or multi_line_comment:
                continue 
            # check if a line contains addParam
            if st:
                if line.strip() != '':
                    # I can expect the invalid addParam call to be the last in the list if it spans multiple lines
                    # based on how I am modifying the list in parse() funciton.
                    all_params[-1] += line.strip()
                    st = False
                else:
                    # skip the white space line, while still keeping the st flag on
                    continue
            else:
                param_statements = regex.finditer(line)
                # print_itr(param_statements)
                all_params, st = parse(line, param_statements)
                if not all_params:
                    continue
            # finds the name of parameters in the current line
            names_of_params_in_line = extract_names_of_params(all_params=all_params, param_map=param_map)
            names_from_comitted_file |= names_of_params_in_line
    
    if DEBUG:
        print('Names from comitted file:')
        print(names_from_comitted_file)
        print('Names from Configuration file:')
        print(names_from_config_file)
    
    # checks if newly added 'addParam' methods are not present in the Configuration.txt
    bad = []
    for x in names_from_comitted_file:
        if x not in names_from_config_file:
            bad.append(x)

    if bad:
        handle_error(bad=bad, param_map=param_map, file_name=file_name)


def parse(line, param_statements):
    """
    Parses a list of param_statements that contain matched 'addParam' instances.
    If the script encounters the '(', it means that addParam call spans multiple lines.
    The function will return in that case.

    :param param_statements: iterator containing the re.Match objects
    :param line: line to parse
    """
    all_params = []
    for match in param_statements:
        method_sig = line[match.start():].strip()
        # check if the addParam will span multiple lines
        stmts = method_sig.split(';')
        for stmt in stmts:
            if stmt != '':
                all_params.append(stmt)
        assert '' not in all_params
        # check if the last addParam call was multi-line
        if all_params[-1][-1] != ')':
            return all_params, True
    return all_params, False


def check_deleted_files(filename):
    """
    Checks if the filename is one of the deleted files. Runs 'git status' and manipulates its output.
    :param filename: name of the file that pre-commit script will pass to the Python script.
    """
    # run 'git status' and split result string by any whitespace
    git_status = subprocess.check_output('git status'.split(' ')).decode(encoding='UTF-8').strip().split()
    # print('Git status: {x}'.format(x=git_status))
    
    # capture all indices of the 'deleted:' in a git_status. 
    deleted_files = [i for i,s in enumerate(git_status) if s == 'deleted:' or s == 'renamed:']
    
    for idx in deleted_files:
        # print('Debug: deleted={x}'.format(x=git_status[idx+1]))
        # the deleted file should be the next index after the idx.
        if filename in git_status[idx + 1]:
            return True
    
    return False
    

def handle_error(bad, param_map, file_name):
    """
    Gets invoked when the 'addParam' from the commit file was not found in the config file.

    :param bad: the list containing the missing 'addParam' names.
    :param param_map: dictionary containing the 'addParam' signature
    :param file_name: the name of the file that was passed in from the bash script.
    """
    print('#' * 20)
    print('File={x} contains the "addParam" statement(s) listed below'.format(x=file_name))
    for item in bad:
        print(param_map[item])
    print('#' * 20)
    sys.exit(1)


def extract_names_of_params(all_params, param_map=None):
    """
    Given the list of parameters, extracts the name of each parameter into a separate list.

    :param all_params: list containing occurences of 'addParam'.
    :param param_map: dictionary to keep track of parameter name and full signature of a parameter.
    :return: list containing names of 'addParam' methods.
    """
    params_within_section = set()
    for param in all_params:
        # big assumption that name of the param will be the first argument
        # it's possible that param will be either a method call, such as addParam(...), or
        # simply formal parameters of the addParam method call, such as "Nx", "abc". The latter
        # happens when the method call spans multiple lines. 
        # This is why we check for '(' and '"'. However, it's also possible that both characters will
        # be present in the param. In that case, we want to grab what comes first. 
        start_ind = min(param.find('(') + 1, param.find('"'))
        if start_ind == -1: start_ind = max(param.find('(') + 1, param.find('"'))
        end_ind = param.find(',') - 1
        name = param[start_ind: end_ind + 1]
        if '"' in name:
            # remove the quotes
            name = name[name.find('"')+1:-1]
        if name != '':
            params_within_section.add(name)
            if param_map is not None:
                param_map[name] = param  # useful when printing
    return params_within_section


def extract_from_config(mapped_section_name):
    """
    Efficiently extracts instances of 'addParam' from the mapped_section_name and GENERAL of the 
    Configuration.txt.

    :param mapped_section_name: the mapped name of the section in the Configuration.txt
    :return: list containing signatures of all 'addParam' instances.
    """
    section = []
    start_parsing = False
    found = 0
    
    # compiling regular expression outside of a loop is more efficient
    regex = re.compile(r'\.*(addParam[^;]+)')

    with open(PATH_TO_CONFIG_FILE) as file:
        st = False # indicates if addParam spans multiple lines
        for line in file:
            line = line.strip() # removes all the whitespace
            if line == BORDER_BEGIN + mapped_section_name or line == BORDER_BEGIN + GENERAL:
                start_parsing = True
                found += 1
                continue
            if line == BORDER_END + mapped_section_name or line == BORDER_END + GENERAL:
                start_parsing = False
                # once I found two sections (i.e. mapped_section_name and GENERAL), I can stop parsing.
                if found == 2:
                    break
                continue
            if start_parsing:
                if st:
                    # will keep skipping lines until the first parameter of addParam occurse
                    # IMPORTANT: if you choose to write multi-line addParamc calls, make sure to 
                    # not have any comments in between the parameters. 
                    if line.strip() != '':
                        in_line_params[-1] += line.strip()
                        # resets the flag
                        st = False
                    else:
                        # skips the empty lines
                        continue
                else:
                    # will return the list of Match objects that will satisfy RegEx pattern in the line
                    in_line_matches = regex.finditer(line)
                    # returns the addParam signatures 
                    in_line_params, st = parse(line, in_line_matches)
                    # skip if line doesn't contain the 'addParam'
                    if not in_line_params:
                        continue
                # adds the signatures to section
                for param in in_line_params:
                    section.append(param)
    # print('Section = {}'.format(section))
    return section


def print_itr(itr):
    """
    Will print values of the iterator if it's not empty. Otherwise, it will state that itr is empty.
    Purely, for debugging purposes. Note, running this function WILL exhaust existing iterator.

    :param: itr: iterable object
    """
    first = next(itr, None)
    if first:
        print('Iterator={} is not empty. Here are its values: '.format(itr))
        print('='*10)
        for x in itr:
            print(x)
        print('='*10)
    else:
        print('Iterator={} is empty.'.format(itr))


def pre_process(file_name):
    """
    Extracts the names of the 'addParam' methods from the Configuration.txt.
    It looks inside the specific and GENERAL sections.

    :param: file_name: the name of the file that was passed in from the bash script.
    :return: the list of all params in the section_name and GENERAL.
    """
    # finds the section name to look for in the Configuration.txt file.
    # the section name is assumed to be the same as the name of the directory where the
    # file_name resides.
    delimiter_pos = [i for i, letter in enumerate(file_name) if letter == '/']

    # edge case is when the edited file is inside the project root directory (e.g. no slashes in the path)
    # or the edited file is only one level down (e.g. themes/Configuration.txt). In both cases, the edited
    # file can not be one of the themes' source code.
    if not delimiter_pos or len(delimiter_pos) == 1:
        return []

    # looks for the text between the first and second slashes.
    # i.e. themes/Jacobi2D/Jacobi2D-Swap.cpp -> extracts Jacobi2D
    section_name = file_name[delimiter_pos[0] + 1: delimiter_pos[1]]
    # print(section_name) 
    # extracts 'addParamInt' signatures from section_name and the GENERAL inside the Configuration.txt
    section = extract_from_config(mapped_section_name=mapping[section_name])
    # finds the names of parameters in the config file
    names_in_section = extract_names_of_params(all_params=section)
    return names_in_section

main()
