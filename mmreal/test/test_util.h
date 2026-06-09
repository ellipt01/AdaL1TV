/**
 * @file test_util.h
 * @brief Minimal dependency-free test harness for the mmreal library.
 *
 * Usage in a test file:
 *
 *   #include "test_util.h"
 *   TEST (my_test) { CHECK (1 + 1 == 2); }
 *   int main (void) {
 *       RUN (my_test);
 *       return test_summary ();   // 0 if all passed, 1 otherwise
 *   }
 */
#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_

#include <stdio.h>
#include <math.h>

static int  g_checks_failed  = 0;   /* failed CHECK count in current test */
static int  g_tests_run      = 0;
static int  g_tests_failed   = 0;

/* Define a test as a void(void) function. */
#define TEST(name)  static void test_##name (void)

/* Run a test, tracking pass/fail. */
#define RUN(name)                                              \
	do {                                                       \
		int _before = g_checks_failed;                         \
		g_tests_run++;                                         \
		printf ("[ RUN  ] %s\n", #name);                       \
		test_##name ();                                        \
		if (g_checks_failed > _before) {                       \
			g_tests_failed++;                                  \
			printf ("[ FAIL ] %s\n", #name);                   \
		} else {                                               \
			printf ("[  OK  ] %s\n", #name);                   \
		}                                                      \
	} while (0)

/* Boolean assertion that does not abort; logs and counts failures. */
#define CHECK(cond)                                            \
	do {                                                       \
		if (!(cond)) {                                         \
			g_checks_failed++;                                 \
			printf ("    CHECK failed: %s  (%s:%d)\n",         \
				#cond, __FILE__, __LINE__);                    \
		}                                                      \
	} while (0)

/* Floating-point near-equality check. */
#define CHECK_NEAR(a, b, tol)                                  \
	do {                                                       \
		double _va = (a), _vb = (b), _t = (tol);               \
		if (!(fabs (_va - _vb) <= _t)) {                       \
			g_checks_failed++;                                 \
			printf ("    CHECK_NEAR failed: %s ~= %s "         \
				"(%g vs %g, tol %g)  (%s:%d)\n",               \
				#a, #b, _va, _vb, _t, __FILE__, __LINE__);     \
		}                                                      \
	} while (0)

/* Print summary and return process exit code. */
static int
test_summary (void)
{
	printf ("\n==== %d tests run, %d failed ====\n",
		g_tests_run, g_tests_failed);
	return (g_tests_failed == 0) ? 0 : 1;
}

#endif /* TEST_UTIL_H_ */
