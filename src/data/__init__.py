from .bugzilla_issue import bugzilla_loader
from .defect import defect_loader
from .github_issue import github_issue_loader
from .static_code import static_code_loader
from .uci import uci_loader


def iterator():
    data_loaders = [uci_loader, defect_loader, bugzilla_loader, github_issue_loader, static_code_loader, uci_loader]

    for loader in data_loaders:
        yield from loader()
