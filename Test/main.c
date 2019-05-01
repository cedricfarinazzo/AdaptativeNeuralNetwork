/*
 * main.c
 *
 */

#include <stdlib.h>
#include <time.h>

#include <criterion/criterion.h>

void setup() {
    srand(time(NULL));
}

int main(int argc, char *argv[]) {
    struct criterion_test_set *tests = criterion_initialize();

    int result = 0;
    if (criterion_handle_args(argc, argv, true))
        result = !criterion_run_all_tests(tests);

    criterion_finalize(tests);
    return result;
}

